import enum


class Provider(enum.StrEnum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    DEEPSEEK = "deepseek"
    VERTEX = "vertex"
    TOGETHER = "together"
    UNIVERSAL_OPENAI = "universal_openai"
