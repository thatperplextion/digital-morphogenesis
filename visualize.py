"""
Produces the two things you actually want for a presentation:
1. growth.png  — a filmstrip of the pattern growing from a single pixel
2. regen.png   — grow it fully, punch a hole in it, keep stepping, watch it heal
"""
import torch
import numpy as np
from PIL import Image
from model import CAModel, make_seed, to_rgb, CHANNEL_N
from train import GRID_SIZE, make_circle_mask

DEVICE = "cpu"


def tensor_to_pil(x, scale=6):
    """x: (1, C, H, W) -> upscaled PIL image for readability."""
    img = to_rgb(x)[0].detach().permute(1, 2, 0).clamp(0, 1).numpy()
    img = (img * 255).astype(np.uint8)
    pil = Image.fromarray(img)
    return pil.resize((pil.width * scale, pil.height * scale), Image.NEAREST)


def make_filmstrip(frames, pad=4):
    w, h = frames[0].size
    strip = Image.new("RGB", (w * len(frames) + pad * (len(frames) - 1), h), (255, 255, 255))
    for i, f in enumerate(frames):
        strip.paste(f, (i * (w + pad), 0))
    return strip


def load_model(path="nca_weights.pt"):
    model = CAModel(channel_n=CHANNEL_N).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


@torch.no_grad()
def growth_sequence(model, n_steps=120, snap_at=(0, 8, 16, 24, 40, 60, 90, 119)):
    x = make_seed(GRID_SIZE, device=DEVICE)
    frames = []
    for step in range(n_steps):
        if step in snap_at:
            frames.append(tensor_to_pil(x))
        x = model(x)
    frames.append(tensor_to_pil(x))
    return make_filmstrip(frames)


@torch.no_grad()
def damage_sequence(model, grow_steps=100, heal_steps=100, snap_every=20):
    x = make_seed(GRID_SIZE, device=DEVICE)
    for _ in range(grow_steps):
        x = model(x)
    frames = [tensor_to_pil(x)]  # fully grown, before damage

    # punch a hole through the middle
    mask = torch.ones(1, 1, GRID_SIZE, GRID_SIZE)
    r = GRID_SIZE * 0.22
    cx = cy = GRID_SIZE / 2
    yy, xx = torch.meshgrid(torch.arange(GRID_SIZE).float(), torch.arange(GRID_SIZE).float(), indexing="ij")
    mask[0, 0] = ((xx - cx) ** 2 + (yy - cy) ** 2 > r ** 2).float()
    x = x * mask
    frames.append(tensor_to_pil(x))  # right after damage

    for step in range(heal_steps):
        x = model(x)
        if (step + 1) % snap_every == 0:
            frames.append(tensor_to_pil(x))
    return make_filmstrip(frames)


@torch.no_grad()
def full_animation_gif(model, path="nca_demo.gif", grow_steps=100, heal_steps=120, damage_at=100):
    """One continuous GIF: grow from seed -> get damaged -> heal. This is the
    file you actually want for a slide (or just play it on loop during your talk)."""
    x = make_seed(GRID_SIZE, device=DEVICE)
    frames = []
    mask = None
    for step in range(grow_steps + heal_steps):
        if step == damage_at:
            mask = torch.ones(1, 1, GRID_SIZE, GRID_SIZE)
            r = GRID_SIZE * 0.22
            cx = cy = GRID_SIZE / 2
            yy, xx = torch.meshgrid(torch.arange(GRID_SIZE).float(), torch.arange(GRID_SIZE).float(), indexing="ij")
            mask[0, 0] = ((xx - cx) ** 2 + (yy - cy) ** 2 > r ** 2).float()
            x = x * mask
        frames.append(tensor_to_pil(x, scale=8))
        x = model(x)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0)
    print(f"saved {path} ({len(frames)} frames)")


if __name__ == "__main__":
    model = load_model()
    growth_sequence(model).save("growth.png")
    damage_sequence(model).save("regen.png")
    full_animation_gif(model)
    print("saved growth.png, regen.png, and nca_demo.gif")
