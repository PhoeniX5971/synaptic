from openai import OpenAI, AsyncOpenAI
from ..openai_.model import OpenAIAdapter
from ...core.capabilities import Capabilities


class DeepSeekAdapter(OpenAIAdapter):
    capabilities = Capabilities()

    @staticmethod
    def _clients(api_key, base_url):
        return (
            OpenAI(api_key=api_key or "none", base_url=base_url or "https://api.deepseek.com"),
            AsyncOpenAI(
                api_key=api_key or "none", base_url=base_url or "https://api.deepseek.com"
            ),
        )
