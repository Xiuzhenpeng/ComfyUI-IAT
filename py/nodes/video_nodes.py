import math
from fractions import Fraction

import torch


_FIXED_FPS = 24


def _audio_sample_rate(audio) -> int:
    sample_rate = int(audio["sample_rate"])
    if sample_rate <= 0:
        raise ValueError("audio sample_rate must be greater than 0")
    return sample_rate


def _audio_sample_count(audio) -> int:
    waveform = audio["waveform"]
    if not torch.is_tensor(waveform):
        waveform = torch.as_tensor(waveform)
    if waveform.numel() == 0 or waveform.dim() == 0:
        return 0
    return int(waveform.shape[-1])


def _frame_count_from_audio(audio, fps: int = _FIXED_FPS) -> int:
    sample_rate = _audio_sample_rate(audio)
    sample_count = _audio_sample_count(audio)
    if sample_count <= 0:
        return 1
    return max(1, int(math.ceil((sample_count / sample_rate) * fps)))


def _first_rgb_image(image: torch.Tensor) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4:
        raise ValueError("image must be a ComfyUI IMAGE tensor with shape [batch, height, width, channels]")
    if image.shape[0] < 1:
        raise ValueError("image batch must contain at least one image")
    if image.shape[-1] < 1:
        raise ValueError("image must contain at least one channel")

    first = image[:1]
    if first.shape[-1] == 1:
        first = first.repeat(1, 1, 1, 3)
    elif first.shape[-1] > 3:
        first = first[..., :3]
    return first.contiguous()


class ImageAudioToVideoNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "create_video"
    CATEGORY = "IAT/Video"
    DESCRIPTION = "Creates a 24 FPS video by repeating one image for the duration of the input audio."

    def create_video(self, image, audio):
        from comfy_api.latest import InputImpl, Types

        frame_count = _frame_count_from_audio(audio, _FIXED_FPS)
        video_frames = _first_rgb_image(image).repeat(frame_count, 1, 1, 1).contiguous()
        components = Types.VideoComponents(
            images=video_frames,
            audio=audio,
            frame_rate=Fraction(_FIXED_FPS, 1),
        )
        return (InputImpl.VideoFromComponents(components),)


NODE_CLASS_MAPPINGS = {
    "ImageAudioToVideo by IAT": ImageAudioToVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAudioToVideo by IAT": "Image + Audio to Video by IAT",
}
