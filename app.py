import huggingface_hub as _hf
if not hasattr(_hf, "HfFolder"):
    class _HfFolder:
        @staticmethod
        def get_token(): return None
        @staticmethod
        def save_token(token): pass
    _hf.HfFolder = _HfFolder

# gradio_client crashes when _json_schema_to_python_type receives a bool schema
# (e.g. additionalProperties: true).  Patch the recursive entry-point so every
# call — including internal recursive ones — bails out early for non-dict input.
import gradio_client.utils as _gcu
_orig_jspt = _gcu._json_schema_to_python_type

def _safe_jspt(schema, *args, **kwargs):
    if not isinstance(schema, dict):
        return "any"
    return _orig_jspt(schema, *args, **kwargs)

_gcu._json_schema_to_python_type = _safe_jspt

import torch
import numpy as np
import gradio as gr
from PIL import Image
from model import build_generator, latent_size

WEIGHTS_FILE = "generator.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = build_generator().to(device)
generator.load_state_dict(torch.load(WEIGHTS_FILE, map_location=device, weights_only=True))
generator.eval()


def denorm(tensor):
    return tensor * 0.5 + 0.5


def generate_faces(num_images: int, seed: int):
    torch.manual_seed(seed)
    with torch.no_grad():
        latent = torch.randn(num_images, latent_size, 1, 1, device=device)
        fake = generator(latent)
        fake = denorm(fake).cpu()

    images = []
    for i in range(num_images):
        arr = fake[i].permute(1, 2, 0).numpy()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        images.append(Image.fromarray(arr))
    return images


with gr.Blocks(title="Anime Face Generator") as demo:
    gr.Markdown(
        """
        # Anime Face Generator
        Generate unique anime faces using a Deep Convolutional GAN trained on 63k anime face images.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            num_slider = gr.Slider(
                minimum=1, maximum=16, value=8, step=1,
                label="Number of faces to generate"
            )
            seed_slider = gr.Slider(
                minimum=0, maximum=9999, value=42, step=1,
                label="Seed (change for different results)"
            )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=3):
            gallery = gr.Gallery(
                label="Generated Anime Faces",
                columns=4,
                rows=2,
                height="auto",
                object_fit="contain"
            )

    generate_btn.click(
        fn=generate_faces,
        inputs=[num_slider, seed_slider],
        outputs=gallery
    )

    demo.load(fn=generate_faces, inputs=[num_slider, seed_slider], outputs=gallery)

    gr.Markdown(
        """
        ---
        Built with PyTorch DCGAN · [Source code](https://github.com/lavkesh1709/Anime_face_generation_through_dcgan)
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
