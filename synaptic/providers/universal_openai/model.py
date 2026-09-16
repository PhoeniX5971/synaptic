from openai import OpenAI, AsyncOpenAI
from ..openai_.model import OpenAIAdapter


class UniversalLLMAdapter(OpenAIAdapter):
    """Chat Completions transport; actual endpoint/model capabilities vary."""

    @staticmethod
    def _clients(api_key, base_url):
        kwargs = {"api_key": api_key or "none", "base_url": base_url}
        return OpenAI(**kwargs), AsyncOpenAI(**kwargs)
