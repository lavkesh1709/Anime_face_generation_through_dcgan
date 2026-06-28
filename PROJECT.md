# Anime Face Generation using DCGAN — Full Technical Write-up

> **Live demo:** [https://huggingface.co/spaces/lavkesh1709/anime-face-generator](https://huggingface.co/spaces/lavkesh1709/anime-face-generator)
> **Source code:** [https://github.com/lavkesh1709/Anime_face_generation_through_dcgan](https://github.com/lavkesh1709/Anime_face_generation_through_dcgan)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset](#2-dataset)
3. [Architecture](#3-architecture)
   - [Generator](#31-generator)
   - [Discriminator](#32-discriminator)
   - [Training Loop](#33-training-loop)
4. [Training Details](#4-training-details)
5. [Web App and API](#5-web-app-and-api)
6. [Deployment Pipeline](#6-deployment-pipeline)
7. [Difficulties and How We Cracked Each One](#7-difficulties-and-how-we-cracked-each-one)
8. [Tech Stack](#8-tech-stack)
9. [Results](#9-results)

---

## 1. Project Overview

This project trains a **Deep Convolutional Generative Adversarial Network (DCGAN)** on 63,000+ anime face images and exposes the trained generator as an interactive web app. Users can generate 1–16 unique anime faces by choosing a seed and clicking a button — the whole inference runs in under a second on CPU.

A GAN consists of two networks playing a minimax game:

- **Generator (G)** — takes a random noise vector and tries to produce an image realistic enough to fool the Discriminator.
- **Discriminator (D)** — tries to tell apart real images (from the dataset) from fake ones (from G).

They are trained simultaneously. As D gets better at spotting fakes, G is forced to produce more convincing images. Equilibrium is reached when G produces images indistinguishable from real data.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source | [Anime Face Dataset — Kaggle](https://www.kaggle.com/datasets/splcher/animefacedataset) |
| Size | ~63,000 images |
| Format | JPEG, RGB |
| Resolution (raw) | Variable |
| Resolution (used) | 64 × 64 px (resized during loading) |

**Preprocessing:**
- All images resized to `64×64`
- Pixel values normalised to `[-1, 1]` using `transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`
- Standard `DataLoader` with `shuffle=True`

The normalisation to `[-1, 1]` is critical because the Generator's final layer is `Tanh` which also outputs `[-1, 1]`. Matching ranges is what makes the loss signal meaningful.

---

## 3. Architecture

### 3.1 Generator

The Generator takes a **100-dimensional latent vector** (random noise sampled from a standard normal distribution) shaped as `(100, 1, 1)` and progressively upsamples it through five `ConvTranspose2d` (transposed convolution / "deconvolution") layers to a `3 × 64 × 64` RGB image.

```
Latent vector: (100, 1, 1)
        ↓  ConvTranspose2d(100→512, k=4, s=1, p=0) + BN + ReLU  →  (512, 4, 4)
        ↓  ConvTranspose2d(512→256, k=4, s=2, p=1) + BN + ReLU  →  (256, 8, 8)
        ↓  ConvTranspose2d(256→128, k=4, s=2, p=1) + BN + ReLU  →  (128, 16, 16)
        ↓  ConvTranspose2d(128→64,  k=4, s=2, p=1) + BN + ReLU  →  (64, 32, 32)
        ↓  ConvTranspose2d(64→3,    k=4, s=2, p=1) + Tanh        →  (3, 64, 64)
Output: RGB image in range [-1, 1]
```

**Key design choices:**
- `bias=False` on every `ConvTranspose2d` — Batch Normalisation already shifts the mean, so a bias term is redundant and wastes parameters.
- `BatchNorm2d` after each layer except the output — stabilises training and prevents mode collapse by keeping activations in a healthy range.
- `ReLU` (not `LeakyReLU`) in the Generator — standard DCGAN convention. ReLU pushes sparse, positive activations which helps G learn sharp features.
- `Tanh` at the output — maps to `[-1, 1]` to match the normalised dataset.

### 3.2 Discriminator

The Discriminator is the mirror image of the Generator — it takes a `3 × 64 × 64` image and collapses it down to a single sigmoid probability (real=1, fake=0) through five `Conv2d` layers.

```
Input: (3, 64, 64)
        ↓  Conv2d(3→64,   k=4, s=2, p=1) + LeakyReLU(0.2)          →  (64, 32, 32)
        ↓  Conv2d(64→128, k=4, s=2, p=1) + BN + LeakyReLU(0.2)     →  (128, 16, 16)
        ↓  Conv2d(128→256,k=4, s=2, p=1) + BN + LeakyReLU(0.2)     →  (256, 8, 8)
        ↓  Conv2d(256→512,k=4, s=2, p=1) + BN + LeakyReLU(0.2)     →  (512, 4, 4)
        ↓  Conv2d(512→1,  k=4, s=1, p=0) + Sigmoid                 →  (1, 1, 1) → scalar
Output: probability ∈ (0, 1)
```

**Key design choices:**
- `LeakyReLU(0.2)` (not ReLU) in the Discriminator — allows a small negative gradient for negative activations, preventing "dead neurons" and helping D learn more robustly.
- No `BatchNorm` on the first layer of D — the input distribution is the data itself; normalising it at this stage destroys the signal D needs to compare real vs fake.
- `Sigmoid` at the output — produces a probability for Binary Cross-Entropy loss.

### 3.3 Training Loop

Each training step processes one batch:

**Step 1 — Train Discriminator:**
```
real_images  → D → real_preds  → BCE(real_preds, ones)   = real_loss
fake_images  → D → fake_preds  → BCE(fake_preds, zeros)  = fake_loss
d_loss = real_loss + fake_loss
d_optimizer.zero_grad(); d_loss.backward(); d_optimizer.step()
```

**Step 2 — Train Generator:**
```
new_fake_images → D → fake_preds → BCE(fake_preds, ones) = g_loss
# G wants D to label its output as REAL (ones), so the label is flipped
g_optimizer.zero_grad(); g_loss.backward(); g_optimizer.step()
```

The crucial trick is **label flipping for G** — instead of minimising `BCE(fake_preds, zeros)` (which saturates and gives vanishing gradients early on), G maximises `log(D(G(z)))` by treating fake images as real. This is the non-saturating formulation from the original GAN paper.

---

## 4. Training Details

| Hyperparameter | Value |
|---|---|
| Latent vector size | 100 |
| Image size | 64 × 64 |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.0002 |
| Adam betas | (0.5, 0.999) |
| Loss function | Binary Cross-Entropy |
| Training platform | Google Colab / Kaggle GPU |

**Why Adam with β₁=0.5?**
The default β₁=0.9 causes GAN training to oscillate badly because momentum carries stale gradients from a rapidly changing loss landscape. Reducing to 0.5 makes the optimizer more reactive to recent gradients, which stabilises the adversarial loop.

**Training history (sample epochs):**

| Epoch | G Loss | D Loss | Real Score | Fake Score |
|---|---|---|---|---|
| 1 | 4.37 | 0.51 | 0.747 | 0.157 |
| 2 | 4.83 | 0.32 | 0.876 | 0.149 |
| 3 | 2.45 | 1.37 | 0.320 | 0.008 |
| 4 | 5.97 | 1.12 | 0.952 | 0.576 |
| 5 | 4.10 | 0.51 | 0.937 | 0.339 |

The oscillation in losses (especially epoch 3 vs 4) is normal GAN behaviour — D and G are constantly adapting to each other.

---

## 5. Web App and API

The inference layer is built with **Gradio** and served via **HuggingFace Spaces (Docker)**.

### Inference flow

```
User sets (num_images, seed)
        ↓
torch.manual_seed(seed)            # reproducible output for same seed
latent = torch.randn(N, 100, 1, 1) # sample N random latent vectors
fake   = generator(latent)         # forward pass through frozen Generator
fake   = tensor * 0.5 + 0.5       # denormalise [-1,1] → [0,1]
images = [PIL.Image from each]     # convert to displayable format
        ↓
Gradio Gallery component
```

The generator weights (`generator.pth`) are loaded once at startup in `eval()` mode with `torch.no_grad()` for the entire lifetime of the server — no reloading per request.

### Gradio UI components

| Component | Role |
|---|---|
| `gr.Slider` (1–16) | Number of faces to generate |
| `gr.Slider` (0–9999) | Random seed for reproducibility |
| `gr.Button` | Trigger generation |
| `gr.Gallery` | Display generated images in a grid |

`demo.load(...)` fires the generation on page load so users see output immediately without having to click.

### Monkey-patches applied at startup

Two compatibility patches are applied at the very top of `app.py` before any Gradio import takes effect:

**Patch 1 — `huggingface_hub.HfFolder` stub:**
Newer versions of `huggingface_hub` removed the `HfFolder` class. Gradio's internals still import it. A minimal stub is injected if it is absent.

**Patch 2 — `gradio_client._json_schema_to_python_type` bool-schema guard:**
Gradio's API introspection parses JSON Schema objects. In some schemas, `additionalProperties` is a plain `bool` (`True`/`False`), which is valid JSON Schema but the parser in `gradio_client 1.0.0` did `if "const" in schema` without checking whether `schema` is actually a dict first — crashing with `TypeError: argument of type 'bool' is not iterable`. The patch wraps the recursive function so any non-dict schema short-circuits to `"any"`.

---

## 6. Deployment Pipeline

```
Local machine
    │
    ├─ git push origin main ──────────────────► GitHub
    │                                           (source of truth)
    │
    └─ python scripts/deploy_to_hf.py ────────► HuggingFace Space
                                                (Docker container)
                                                     │
                                             pip install requirements.txt
                                             python app.py
                                             Gradio on 0.0.0.0:7860
                                             HF reverse-proxy → public URL
```

**Dockerfile summary:**
- Base: `python:3.11` (3.12 dropped `audioop` which `pydub`/`gradio` depended on)
- Non-root user (`user` at UID 1000) — HF Spaces requirement
- `EXPOSE 7860` + `ENV GRADIO_SERVER_NAME=0.0.0.0`
- Entrypoint: `python app.py`

---

## 7. Difficulties and How We Cracked Each One

This section documents every blocker hit during development and the exact fix applied.

---

### 7.1 `audioop` crash on Python 3.12

**Error:**
```
ModuleNotFoundError: No module named 'audioop'
```
**Cause:** Python 3.12 removed the built-in `audioop` module. Gradio depends on `pydub` which imports `audioop`.

**Fix:** Pin the Docker base image to `python:3.11` in `Dockerfile`. Python 3.11 still includes `audioop`.

---

### 7.2 `HfFolder` import error

**Error:**
```
ImportError: cannot import name 'HfFolder' from 'huggingface_hub'
```
**Cause:** `huggingface_hub >= 0.24` removed the legacy `HfFolder` class. Gradio 5's internals still tried to import it.

**Fix:** Inject a minimal stub at the top of `app.py` before any Gradio import:
```python
import huggingface_hub as _hf
if not hasattr(_hf, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(token): pass
    _hf.HfFolder = _HfFolder
```

---

### 7.3 Jinja2 version conflict

**Error:**
```
ImportError: cannot import name 'Markup' from 'jinja2'
```
**Cause:** `Jinja2 >= 3.1` removed `Markup` from the top-level namespace. An old transitive dependency of Gradio still used the old import path.

**Fix:** Pinned `Jinja2<3.1` in `requirements.txt` temporarily, then resolved by upgrading Gradio to a version whose dependency tree no longer had the conflict.

---

### 7.4 `gradio_client` bool-schema `TypeError`

**Error:**
```
TypeError: argument of type 'bool' is not iterable
  File "gradio_client/utils.py", line 882, in get_type
    if "const" in schema:
```
**Cause:** JSON Schema allows `additionalProperties: true` (a plain boolean). The `_json_schema_to_python_type` function in `gradio_client 1.0.0` did not check `isinstance(schema, dict)` before doing `"const" in schema`, so when called recursively with a `bool`, it crashed.

**First attempt (wrong):** Patching `_gcu.get_type` — failed because `get_type` is a module-level function and its reference inside `_json_schema_to_python_type` is resolved at call time through the module's global namespace. In theory this should work, but `get_type` being called from within the recursive closure of the same function means the patch was not reaching deep recursive calls reliably.

**Working fix:** Patch `_json_schema_to_python_type` itself — the recursive entry-point:
```python
import gradio_client.utils as _gcu
_orig = _gcu._json_schema_to_python_type

def _safe(schema, *args, **kwargs):
    if not isinstance(schema, dict):
        return "any"
    return _orig(schema, *args, **kwargs)

_gcu._json_schema_to_python_type = _safe
```
Because recursive calls from within `_orig` look up `_json_schema_to_python_type` in the module's global dict (which now points to `_safe`), the guard applies to every level of recursion automatically.

---

### 7.5 `ValueError: When localhost is not accessible`

**Error:**
```
ValueError: When localhost is not accessible, a shareable link must be created.
Please set share=True or check your proxy settings to allow access to localhost.
```
**Cause:** Gradio 5.0.0 had a bug where it used `server_name` as the host in its internal health-check URL. When `server_name="0.0.0.0"`, the URL became `http://0.0.0.0:7860/` — which is not a routable address you can connect *to*. The connectivity test failed, Gradio concluded the server wasn't accessible, and raised the error.

**Fix:** Upgraded `gradio==5.0.0` → `gradio>=5.5.0,<6.0.0`. In Gradio 5.3+, the internal health-check always uses `http://127.0.0.1:{port}/` regardless of what `server_name` is set to, so binding to `0.0.0.0` works correctly.

---

### 7.6 `theme` parameter deprecation crash

**Error:**
```
TypeError: Blocks.launch() got an unexpected keyword argument 'theme'
```
**Cause:** Misread the deprecation warning `"theme parameter in Blocks() will be removed in Gradio 6"` as "move it to `launch()`". In Gradio 5.x, `theme` is *only* valid in the `Blocks()` constructor; `launch()` does not accept it.

**Fix:** Reverted `theme` back to `gr.Blocks(title=..., theme=gr.themes.Soft())`. The deprecation warning is informational only — the constructor still works fine throughout the entire Gradio 5.x series.

---

## 8. Tech Stack

| Layer | Technology |
|---|---|
| Model framework | PyTorch 2.x |
| Image processing | Pillow, NumPy |
| Web UI | Gradio 5.5+ |
| HF Hub integration | huggingface_hub |
| Containerisation | Docker (python:3.11 base) |
| Hosting | HuggingFace Spaces (Docker SDK) |
| Source control | GitHub |
| Training environment | Google Colab / Kaggle (GPU) |

---

## 9. Results

- Generator produces `64×64` RGB anime faces
- Seed control gives reproducible outputs — same seed always yields the same batch
- Inference time: < 1 second on CPU for up to 16 images
- Model weights: `generator.pth` at ~14 MB

The Discriminator weights (`discriminator.pth`) are saved but not needed for inference — only the Generator is loaded in the web app.
