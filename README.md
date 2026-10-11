# Growing Neural Cellular Automata

| | |
|---|---|
| **Stack** | Python, PyTorch, NumPy, Pillow, Matplotlib |
| **Type** | Deep learning, reimplementation of a published paper |
| **Output** | [`growth.png`](growth.png), [`regen.png`](regen.png), [`nca_demo.gif`](nca_demo.gif) |

A tiny neural network that learns a *local update rule*: apply it to every
pixel of a grid, over and over, and a target image grows itself from a
single seed pixel — and if you damage it mid-growth, it heals.

Based on: Mordvintsev, Randazzo, Niklasson, Levin, "Growing Neural Cellular
Automata", Distill, 2020. https://distill.pub/2020/growing-ca/

## Why this project

- It's a real, published architecture with a clear story (perception →
  learned update → stochastic async application → life masking) — every
  design choice has a reason you can explain on a slide.
- The "dataset" is one image. No scraping, no labeling, no data pipeline.
- The demo is genuinely impressive live: grow it, then damage it in front
  of the audience and watch it heal in real time.
- It has an obvious, satisfying growth path after the course (see below).

## Run

```bash
pip install -r requirements.txt
python make_target.py          # draw the target image
python train.py 1000           # train; writes nca_weights.pt and loss_log.npy
python visualize.py           # writes growth.png, regen.png, nca_demo.gif
```

## Files

| File | What it does |
|---|---|
| `make_target.py` | Draws the target image (a star) with PIL — swap this for any RGBA PNG |
| `model.py` | The `CAModel`: fixed Sobel perception + tiny learned update MLP |
| `train.py` | Training loop: sample pool + damage, which is what makes regeneration emerge |
| `visualize.py` | Produces `growth.png` (growth filmstrip) and `regen.png` (damage → heal filmstrip) |

## How it works

1. **State**: every cell is a 16-number vector. Channels 0-3 are RGBA
   (visible). Channels 4-15 are hidden — the cell's private "memory," which
   the network invents its own use for during training. Nobody tells it
   what to put there.
2. **Perception**: each cell looks at itself and its 8 neighbors through
   three *fixed, non-learned* filters — an identity filter and the two
   Sobel edge filters (x-gradient, y-gradient). This is the only channel of
   information between cells. It's fixed on purpose: it forces the network
   to learn everything useful in the *update* step, not in a smarter
   perception step, which keeps the model a genuine "local rule" rather
   than something that's secretly doing global convolutions.
3. **Update rule**: a 2-layer MLP (implemented as two 1×1 convolutions)
   takes each cell's perception vector and outputs a delta to add to that
   cell's state. This is the *only* learned part of the model — a few
   thousand parameters, applied identically to every cell.
4. **Stochastic + masked update**: each cell only applies its update ~50%
   of the time (independently), and any cell whose neighborhood alpha is
   too low gets zeroed out. Together these stop the pattern from either
   syncing up into a rigid clock or leaking into empty space forever.
5. **Training**: run the rule for a random number of steps (64-96) from a
   pool of grids at various stages of growth (some fresh, most partially
   or fully grown, some randomly damaged), compare the RGBA channels to
   the target image, backprop through the whole unrolled sequence. Repeat
   thousands of times.

Regeneration is never explicitly rewarded — it falls out of the training
distribution containing damaged states. The key point: **the model isn't told to heal, it's
just never allowed to see "damaged" as an excuse for high loss.**

## Possible extensions

1. **Localize damage response** — visualize the hidden channels directly
   (they're not RGB, but you can PCA them down to 3 channels and render
   them as false color) to show the "wound signal" propagating.
2. **Multiple stable patterns from one model** — condition the update MLP
   on a one-hot "which pattern" input so a single trained network can grow
   several different images depending on a control signal.
3. **3D**: extend the grid to voxels and the perception filters to 3D
   Sobel kernels; grow a simple 3D shape instead of a 2D image.
4. **Interactive browser demo**: export the trained weights and re-implement
   the forward pass (it's just two convolutions) in plain JavaScript/WebGL
   so anyone can click to damage it live in a browser with zero install —
   this is exactly how the original Distill.pub article's demos work.
5. **Robustness study**: systematically vary damage size/shape/timing during
   training and measure how regeneration quality degrades — this turns the
   project into an actual research-style ablation study, good for a final
   report or a portfolio write-up.

## Honest caveats

- The official implementation trains for ~8000 iterations; anything trained
  for much less will show growth but weaker, slower regeneration. Train
  longer than the demo checkpoint included here if you want polished results.
- CPU training is slow (~1 it/s at this grid size on a single core). Use a
  free Colab GPU for the real run — it should be 10-20x faster.
