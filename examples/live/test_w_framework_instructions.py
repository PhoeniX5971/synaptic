from synaptic.core import Model, Provider, Tool, autotool
from dotenv import load_dotenv
import os

# load .env file
load_dotenv()

# fetch key from env
api_key = os.getenv("GEMINI_API_KEY", "")

instructions = """
You are Memizer.
You are a helpful model that likes to throw a lot of jokes.
Be funny with your outputs.
"""


model = Model(
    provider=Provider.GEMINI,
    model="gemini-2.5-flash",
    temperature=1,
    api_key=api_key,
    instructions=instructions,
    automem=True,
    autorun=True,
)

prompt = """
Instroduce yourself based on your instructions.
    """

memory = model.invoke(prompt, role="user")
print(
    memory.message
    + "\n\n\n"
    + str(memory.created)
    + "\n\n\n"
    + str(memory.tool_calls)
    + "\n\n\n"
    + str(memory.tool_results)
    + "\n\n\n"
    + str(model.history.MemoryList)
)
