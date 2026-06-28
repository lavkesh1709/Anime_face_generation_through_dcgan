---
title: Anime Face Generator
emoji: 🎌
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Anime Face Generation using DCGAN

Generate anime faces using a Deep Convolutional GAN (DCGAN) trained on 63,000+ anime face images.

## Live Demo

[Hugging Face Spaces](https://huggingface.co/spaces/lavkesh1709/anime-face-generator) *(deploy to activate)*

## Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/lavkesh1709/Anime_face_generation_through_dcgan.git
cd Anime_face_generation_through_dcgan

# 2. Install dependencies
pip install torch torchvision gradio Pillow numpy

# 3. Launch the app
python app.py
```

Then open `http://localhost:7860` in your browser.

## Project Structure

```
├── app.py                  # Gradio web UI
├── model.py                # Generator architecture
├── generator.pth           # Trained generator weights
├── discriminator.pth       # Trained discriminator weights
├── requirements.txt        # Dependencies for HF Spaces
├── training_history.csv    # Loss/score per epoch
└── Anime_face_generation_through_dcgan.ipynb  # Training notebook (Google Colab)
```

## Architecture

**Generator** — takes a 100-dim random latent vector and upsamples through 5 ConvTranspose2d layers to produce a 64×64 RGB image.

**Discriminator** — classifies real vs fake images through 5 Conv2d layers down to a single sigmoid output.

Both trained with Adam optimizer (lr=0.0002, betas=(0.5, 0.999)) and binary cross-entropy loss.

## Dataset

[Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) — 63,000+ anime face images from Kaggle.

## Acknowledgements

Built with PyTorch. Dataset from Kaggle.
