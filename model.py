"""
Growing Neural Cellular Automata.
Based on Mordvintsev et al., "Growing Neural Cellular Automata", Distill 2020.
https://distill.pub/2020/growing-ca/

The core idea:
- Every pixel ("cell") is a vector of CHANNEL_N numbers, not just RGB.
  Channels 0-3 are RGBA (what we can see). Channels 4+ are "hidden state" —
  the cell's private memory, which it learns to use for communicating with
  its neighbors.
- At every step, EVERY cell runs the exact same tiny neural network,
  looking only at itself and its immediate neighbors (via fixed Sobel
  filters, not learned ones — this is what makes it a real local rule
  rather than a global image generator).
- Repeat that update thousands of times and complex, self-organizing
  images emerge. Train it while randomly damaging cells mid-growth and
  it learns to regenerate, because the rule has to work no matter what
  state its neighbors happen to be in.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

CHANNEL_N = 16      # 4 visible (RGBA) + 12 hidden "memory" channels
CELL_FIRE_RATE = 0.5  # async update: only ~50% of cells update each step


def make_seed(size, channel_n=CHANNEL_N, n=1, device="cpu"):
    """A single living pixel in the center of an otherwise empty grid."""
    seed = torch.zeros(n, channel_n, size, size, device=device)
    seed[:, 3:, size // 2, size // 2] = 1.0  # alpha + all hidden channels = 1 at the center
    return seed


def get_living_mask(x):
    """A cell is 'alive' if it or a neighbor has alpha > 0.1.
    This is what lets empty space stay empty forever instead of the
    whole grid slowly filling with noise."""
    alpha = x[:, 3:4, :, :]
    return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1


class CAModel(nn.Module):
    def __init__(self, channel_n=CHANNEL_N, hidden_n=128):
        super().__init__()
        self.channel_n = channel_n

        # --- Perception: fixed (non-learned) filters, one identity + two Sobel ---
        # This is the ONLY way information crosses cell boundaries. Everything
        # else the network learns is purely about what to DO with what it perceives.
        identity = torch.zeros(3, 3)
        identity[1, 1] = 1.0
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 8.0
        sobel_y = sobel_x.t()
        kernel = torch.stack([identity, sobel_x, sobel_y], dim=0).unsqueeze(1)  # (3,1,3,3)
        kernel = kernel.repeat(channel_n, 1, 1, 1)  # (3*channel_n, 1, 3, 3) — depthwise, per-channel
        self.register_buffer("perception_kernel", kernel)

        # --- Update rule: this is the ONLY part of the model with learned weights ---
        # It's just a 2-layer MLP applied identically, independently, to every cell
        # (implemented as 1x1 convs, which is exactly the same thing).
        self.update_net = nn.Sequential(
            nn.Conv2d(channel_n * 3, hidden_n, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_n, channel_n, kernel_size=1, bias=False),
        )
        # Zero-init the last layer: at the start of training the update does
        # nothing, so training starts from a stable "do nothing" rule rather
        # than immediately exploding.
        nn.init.zeros_(self.update_net[-1].weight)

    def perceive(self, x):
        return F.conv2d(x, self.perception_kernel, padding=1, groups=self.channel_n)

    def forward(self, x, fire_rate=CELL_FIRE_RATE):
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x)
        dx = self.update_net(y)

        # stochastic update: each cell independently decides whether to
        # apply its update this step. This isn't a training trick — it
        # forces the rule to be robust to cells being out of sync with
        # each other, which is what makes it look organic instead of
        # like a simple diffusion.
        update_mask = (torch.rand(x[:, :1, :, :].shape, device=x.device) <= fire_rate).float()
        x = x + dx * update_mask

        post_life_mask = get_living_mask(x)
        life_mask = (pre_life_mask & post_life_mask).float()
        return x * life_mask  # cells that died (or never lived) are zeroed out


def to_rgb(x):
    """Composite the RGBA channels onto a white background for display."""
    rgb, a = x[:, :3, :, :], torch.clamp(x[:, 3:4, :, :], 0, 1)
    return 1.0 - a + rgb  # white background, alpha-blended
