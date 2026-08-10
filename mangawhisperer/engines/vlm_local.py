"""Plan-B Vision-Language engine: local Qwen-VL, no API cost.

Runs a Qwen2.5-VL-class open model (default 7B, 4-bit quantized so it
fits a 16 GB T4 on Colab) with the same scriptwriter system prompt as
the Claude engine, and validates the model's JSON output against the
same :class:`ContextualizedBlock` contract. Trade-off vs the Claude
engine: zero per-panel cost, but weaker PT-BR action descriptions and
diarization.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Callable, Sequence

from mangawhisperer.engines.script_parsing import parse_script_blocks, passthrough_blocks
from mangawhisperer.engines.vlm import DEFAULT_CAST, build_scriptwriter_prompt
from mangawhisperer.interfaces import Image, VisionLanguageEngine
from mangawhisperer.models import ContextualizedBlock, SpeechBubble

logger = logging.getLogger(__name__)

Generator = Callable[[Image, str], str]
"""Takes (panel image, user prompt), returns the model's raw text."""

_JSON_INSTRUCTION = """

Responda APENAS com um array JSON, sem markdown e sem texto extra, no formato:
[{"text": "...", "speaker_id": "...", "is_speech": true}, ...]
Nunca inclua blocos com "text" vazio — simplesmente omita-os.
No máximo 2 blocos de ação (is_speech=false) por painel.\
"""


class QwenVisionLanguageEngine(VisionLanguageEngine):
    """Local open-source scriptwriter (Qwen2.5-VL via transformers).

    The model loads lazily on first use; tests inject a fake
    ``generator`` so no weights are needed. Malformed model output
    degrades to unattributed passthrough blocks instead of aborting.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        known_characters: Sequence[str] = DEFAULT_CAST,
        sfx_tags: Sequence[str] = (),
        sfx_intensity: int = 2,
        style_addendum: str = "",
        quantize_4bit: bool = True,
        max_new_tokens: int = 2048,
        max_image_pixels: int = 1280 * 28 * 28,
        generator: Generator | None = None,
    ) -> None:
        """
        Args:
            model_name: Hugging Face model id. Drop to
                ``Qwen/Qwen2.5-VL-3B-Instruct`` if the GPU is tight.
            known_characters: Cast names for speaker attribution.
            quantize_4bit: Load in 4-bit (needs bitsandbytes); makes
                the 7B model fit a 16 GB T4.
            max_new_tokens: Generation budget per panel.
            max_image_pixels: Vision-token cap passed to the processor.
            generator: Injectable ``(image, prompt) -> text`` callable
                for tests; ``None`` builds the real pipeline lazily.
        """
        self._model_name = model_name
        self._quantize_4bit = quantize_4bit
        self._max_new_tokens = max_new_tokens
        self._max_image_pixels = max_image_pixels
        self._generator = generator
        self._model: Any = None
        self._processor: Any = None
        self._system_prompt = (
            build_scriptwriter_prompt(known_characters, sfx_tags, sfx_intensity, style_addendum)
            + _JSON_INSTRUCTION
        )

    @property
    def fingerprint(self) -> str:
        """Checkpoint identity: local model + prompt define the output."""
        digest = hashlib.sha1(self._system_prompt.encode("utf-8")).hexdigest()[:8]
        return f"qwen-local:{self._model_name}:prompt={digest}"

    def contextualize(
        self, panel_image: Image, bubbles: list[SpeechBubble]
    ) -> list[ContextualizedBlock]:
        """Produce the ordered narration script for one panel."""
        request_text = (
            "Textos das bolhas de fala, em ordem de leitura (lista JSON):\n"
            + json.dumps([b.text for b in bubbles], ensure_ascii=False)
        )
        raw = (self._generator or self._generate)(panel_image, request_text)

        blocks = parse_script_blocks(raw)
        if blocks is None or (not blocks and bubbles):
            logger.warning(
                "Qwen-VL output yielded no usable blocks (%r...); falling back to "
                "passthrough for %d bubbles",
                raw[:80],
                len(bubbles),
            )
            return passthrough_blocks(bubbles)
        return blocks

    def release(self) -> None:
        """Free the model's VRAM (used between pipeline stages so the
        TTS model fits alongside on small GPUs)."""
        if self._model is None:
            return
        self._model = None
        self._processor = None
        try:
            import torch  # noqa: PLC0415

            torch.cuda.empty_cache()
        except Exception:
            pass
        logger.info("Qwen-VL model released from memory")

    def _generate(self, panel_image: Image, request_text: str) -> str:
        """Run the real Qwen-VL model (loaded lazily on first call)."""
        import torch  # noqa: PLC0415 — heavy imports, deferred until needed
        from PIL import Image as PILImage  # noqa: PLC0415

        model, processor = self._get_model()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self._system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": request_text},
                ],
            },
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt], images=[PILImage.fromarray(panel_image)], return_tensors="pt"
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=self._max_new_tokens, do_sample=False
            )
        trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def _get_model(self) -> tuple[Any, Any]:
        if self._model is None:
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )

            logger.info("Loading %s (first run downloads the weights)", self._model_name)
            model_kwargs: dict[str, Any] = {"torch_dtype": "auto", "device_map": "auto"}
            if self._quantize_4bit:
                try:
                    from transformers import BitsAndBytesConfig  # noqa: PLC0415

                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                    )
                except Exception:
                    logger.warning("bitsandbytes unavailable; loading unquantized")

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self._model_name, **model_kwargs
            )
            self._processor = AutoProcessor.from_pretrained(
                self._model_name, max_pixels=self._max_image_pixels
            )
        return self._model, self._processor
