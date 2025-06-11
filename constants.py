DEFAULT_MODEL = "gpt-4o"
DEFAULT_FALLBACK_MODEL = "claude-3-7-sonnet-latest"
DEFAULT_REASONING_MODEL = "o1"
DUMP_RAW_LLM_OUTPUT = False


# Much cheaper, but doesn't seem to be able to handle function calls very well
# DEFAULT_REALTIME_AUDIO_MODEL = "gpt-4o-mini-realtime-preview-2024-12-17"

DEFAULT_REALTIME_AUDIO_MODEL = "gpt-4o-realtime-preview"

DEFAULT_REALTIME_AUDIO_VOICE = "shimmer"

# Mode constants
REALTIME_AUDIO_CONVERSATION = "REALTIME_AUDIO_CONVERSATION"
TEXT = "TEXT"  # This mode is for text-based conversations, but is less cool and not maintained

# Set the mode
MODE = REALTIME_AUDIO_CONVERSATION
