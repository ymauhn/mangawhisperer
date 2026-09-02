"""Leaf module with the pipeline-wide audio invariants.

Kept dependency-free so any engine can import it without pulling the
orchestrator/config graph (``config`` re-exports these for callers).
Every narration segment, effect and music bed is normalized to this
format before mixing.
"""

AUDIO_SAMPLE_RATE = 24_000  # Hz — XTTSv2's native output rate
AUDIO_CHANNELS = 1  # mono
AUDIO_SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM
