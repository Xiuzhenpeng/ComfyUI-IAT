"""Audio helper nodes for ComfyUI-IAT."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image


def _first_image_to_pil(image: torch.Tensor) -> Image.Image:
    """Convert the first ComfyUI IMAGE tensor in a batch to an RGB PIL image."""
    if not isinstance(image, torch.Tensor):
        raise TypeError("image must be a torch.Tensor")

    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[-1] not in (1, 3, 4):
        raise ValueError("image must have shape [batch, height, width, channels]")

    arr = (image[0].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    pil_image = Image.fromarray(arr)
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def _image_to_png_bytes(image: torch.Tensor) -> bytes:
    """Serialize the first ComfyUI IMAGE tensor in a batch as PNG bytes."""
    pil_image = _first_image_to_pil(image)
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


class AudioSetCoverNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "set_cover"
    CATEGORY = "IAT/Audio"
    DESCRIPTION = (
        "Attach the first input image as cover-art metadata on a ComfyUI AUDIO object "
        "without changing the waveform."
    )

    def set_cover(self, image: torch.Tensor, audio: dict[str, Any]):
        if not isinstance(audio, dict):
            raise TypeError("audio must be a dict-like ComfyUI AUDIO object")
        if "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError("audio must contain 'waveform' and 'sample_rate'")

        cover_png = _image_to_png_bytes(image)
        output_audio = dict(audio)
        output_audio["cover_image"] = cover_png
        output_audio["cover_image_mime_type"] = "image/png"
        output_audio["cover_image_format"] = "png"
        output_audio["cover_image_base64"] = base64.b64encode(cover_png).decode("ascii")
        return (output_audio,)


NODE_CLASS_MAPPINGS = {
    "AudioSetCoverNode by IAT": AudioSetCoverNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioSetCoverNode by IAT": "Audio Set Cover by IAT",
}
