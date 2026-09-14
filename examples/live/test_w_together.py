import os
import time

from dotenv import load_dotenv

from synaptic import Model, Provider

load_dotenv()

api_key = os.getenv("TOGETHER_API_KEY")

llm = Model(
    provider=Provider.TOGETHER,
    model="Qwen/Qwen3-VL-8B-Instruct",
    api_key=api_key,  # type: ignore
)

start = time.perf_counter()

response = llm.invoke(
    prompt="What do you see in the image.",
    images=["docs/assets/20260314222926.png"],
)

end = time.perf_counter()

print(response.message)
print(f"\nLatency: {end - start:.3f} seconds")
