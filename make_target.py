"""
Generates the target image the CA will learn to grow into.
No internet needed — we draw it ourselves with PIL.
Feel free to swap this out for your own emoji/logo/PNG later
(just make sure it has a transparent (RGBA) background).
"""
import numpy as np
from PIL import Image, ImageDraw


def make_target(size=32):
    """Draws a simple 5-pointed star with a soft gradient fill on a transparent background."""
    # draw at 4x resolution then downsample -> gives us anti-aliased edges,
    # which matters a lot for this model (it learns from the alpha gradient)
    hi_res = size * 4
    img = Image.new("RGBA", (hi_res, hi_res), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx, cy = hi_res / 2, hi_res / 2
    r_outer = hi_res * 0.42
    r_inner = r_outer * 0.42
    points = []
    for i in range(10):
        angle = -np.pi / 2 + i * np.pi / 5
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * np.cos(angle), cy + r * np.sin(angle)))

    # simple orange->pink gradient fill by drawing many concentric-ish layers
    draw.polygon(points, fill=(255, 140, 40, 255))
    # overlay a lighter core for a bit of shading
    inner_points = [(cx + 0.55 * (x - cx), cy + 0.55 * (y - cy)) for x, y in points]
    draw.polygon(inner_points, fill=(255, 210, 90, 255))

    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0  # (H, W, 4), values in [0,1]
    return arr


if __name__ == "__main__":
    target = make_target(32)
    print("target shape:", target.shape, "alpha range:", target[..., 3].min(), target[..., 3].max())
    Image.fromarray((target * 255).astype(np.uint8)).save("target.png")
    print("saved target.png")
