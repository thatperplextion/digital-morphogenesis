"""
Training loop for the Growing Neural Cellular Automata.

Two things make this more than "just a video prediction model":

1. SAMPLE POOL: instead of always starting every training step from a fresh
   seed, we keep a pool of ~256 grids that are further along in growth
   (including some that grew into the full target and kept running).
   We train on THOSE too. This is what makes the pattern learn to be
   *stable* forever, not just correct at one specific timestep — a plain
   "grow for N steps" model tends to overshoot into noise once it's already
   correct, because it never saw examples of "already correct, stay put."

2. DAMAGE: some fraction of the pool gets a random chunk erased before the
   next training step. The model is trained to fix it. It never sees
   "damage" as a special case in the loss function — it just has to keep
   minimizing pixel error, and the only way to do that after damage is to
   regrow. Regeneration is a side-effect of the objective, not something
   we coded by hand.
"""
import torch
import torch.nn.functional as F
import numpy as np
from model import CAModel, make_seed, to_rgb, CHANNEL_N
from make_target import make_target

torch.manual_seed(0)
np.random.seed(0)

GRID_SIZE = 32
POOL_SIZE = 256
BATCH_SIZE = 8
DEVICE = "cpu"


def make_circle_mask(size, device):
    """Random circular 'damage' mask used to erase part of a grown grid."""
    x = torch.arange(size, device=device).float()
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    cx, cy = np.random.uniform(0.2, 0.8) * size, np.random.uniform(0.2, 0.8) * size
    r = np.random.uniform(0.1, 0.25) * size
    return ((xx - cx) ** 2 + (yy - cy) ** 2 > r ** 2).float()  # 1 outside the circle, 0 inside


def train(n_iters=1000, log_every=50, ckpt_every=100, ckpt_path="nca_weights.pt",
          log_path="loss_log.txt", resume_from=None, lr=2e-3):
    target_np = make_target(GRID_SIZE)  # (H, W, 4)
    target = torch.tensor(target_np).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1,4,H,W)

    model = CAModel(channel_n=CHANNEL_N).to(DEVICE)
    if resume_from is not None:
        model.load_state_dict(torch.load(resume_from, map_location=DEVICE))
        print(f"resumed weights from {resume_from}")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(n_iters * 0.7)], gamma=0.1)

    seed = make_seed(GRID_SIZE, device=DEVICE)  # (1, C, H, W)
    pool = seed.repeat(POOL_SIZE, 1, 1, 1).clone()

    loss_log = []
    log_f = open(log_path, "w")

    for it in range(n_iters):
        idx = np.random.choice(POOL_SIZE, BATCH_SIZE, replace=False)
        x = pool[idx].clone()

        # rank this batch by current loss, replace the WORST one with a
        # fresh seed -> keeps the model from forgetting how to grow from
        # scratch, since most of the pool will otherwise already be "grown"
        with torch.no_grad():
            batch_losses = ((x[:, :4] - target) ** 2).mean(dim=[1, 2, 3])
            worst = batch_losses.argmax().item()
            x[worst] = seed[0]

            # damage ~half the batch (skip the fresh-seed slot)
            for i in range(BATCH_SIZE):
                if i != worst and np.random.rand() < 0.5:
                    mask = make_circle_mask(GRID_SIZE, DEVICE)
                    x[i] = x[i] * mask

        n_steps = np.random.randint(64, 96)
        for _ in range(n_steps):
            x = model(x)

        loss = F.mse_loss(x[:, :4], target.expand(BATCH_SIZE, -1, -1, -1))
        opt.zero_grad()
        loss.backward()
        for p in model.parameters():
            if p.grad is not None:
                p.grad = p.grad / (p.grad.norm() + 1e-8)  # gradient normalization (per original paper)
        opt.step()
        sched.step()

        pool[idx] = x.detach()
        loss_log.append(loss.item())

        if it % log_every == 0 or it == n_iters - 1:
            msg = f"iter {it:5d}  loss {loss.item():.5f}"
            print(msg, flush=True)
            log_f.write(msg + "\n")
            log_f.flush()

        if it % ckpt_every == 0 or it == n_iters - 1:
            torch.save(model.state_dict(), ckpt_path)
            np.save("loss_log.npy", np.array(loss_log))

    log_f.close()
    return model, target, loss_log


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("n_iters", type=int, nargs="?", default=1000)
    p.add_argument("--resume", type=str, default=None, help="path to weights to resume from")
    p.add_argument("--lr", type=float, default=2e-3)
    args = p.parse_args()

    model, target, loss_log = train(n_iters=args.n_iters, resume_from=args.resume, lr=args.lr)
    torch.save(model.state_dict(), "nca_weights.pt")
    np.save("loss_log.npy", np.array(loss_log))
    print("saved nca_weights.pt and loss_log.npy")


