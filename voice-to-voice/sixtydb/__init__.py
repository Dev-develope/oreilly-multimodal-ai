"""60dB AI provider plugin for LiveKit Agents.

Provides STT, LLM and TTS implementations that follow the same plugin
pattern as the official ``livekit.plugins.{openai,deepgram}`` packages, so
they can be dropped straight into a ``VoicePipelineAgent``:

    from sixtydb import STT, LLM, TTS

    agent = VoicePipelineAgent(
        stt=STT(),
        llm=LLM(model="60db-chat"),
        tts=TTS(voice="andrew"),
        ...
    )

Configuration is read from the environment (see ``.env.example``):

    SIXTYDB_API_KEY      required - bearer token for the 60dB API
    SIXTYDB_BASE_URL     optional - defaults to https://api.60db.ai/v1

IMPORTANT: 60dB's request/response wire format is not publicly documented
here. The request/response shapes in stt.py / llm.py / tts.py are the
project's best-guess defaults (OpenAI-style JSON + SSE). If 60dB's real API
differs, adjust the marked ``# 60dB API:`` sections - the LiveKit-facing
contract (the classes below) does not need to change.
"""

from .stt import STT
from .llm import LLM
from .tts import TTS
from .config import SixtyDBConfig

__all__ = ["STT", "LLM", "TTS", "SixtyDBConfig"]
