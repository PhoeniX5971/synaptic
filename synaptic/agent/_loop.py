"""The one agent loop. It decides; a driver performs the model call and tool effects.

`turns` is a plain generator so the same control flow serves the synchronous driver
(`Agent.run`) and the streaming driver (`Agent.astream`; `Agent.arun` drains it).
It yields ("model", role) and expects (response, final_seen) back, then yields
("tools", response) and expects the driver to have dispatched that response's calls.
"""

from ..core._execution_state import check_cancelled
from .session import BudgetExceeded

PROTOCOL_REPROMPT = (
    "No <tool> or <final> emitted. Emit <tool> to continue or <final></final> to finish."
)


def turns(agent, prompt, role, images=None, audio=None, abort=None):
    agent._add_user(prompt, role, images, audio)
    retries, agent._final = 0, None
    agent._outbox.clear()
    for _ in range(agent.max_turns):
        check_cancelled(abort)
        agent._prepare_turn()
        memory, final_seen = yield ("model", role)
        agent.session.history.add(memory)
        agent.pending_calls = tuple(memory.tool_calls)
        yield ("tools", memory)
        role = "user"
        if agent._conclude(memory):  # A tool asked to end the run with a result.
            return memory
        if memory.tool_calls:
            retries = 0
            continue
        if agent._text_mode and agent.signals and not final_seen:
            if retries >= agent.max_protocol_retries:
                raise BudgetExceeded("Agent reached max_protocol_retries")
            retries += 1
            agent._add_user(PROTOCOL_REPROMPT, "user")
            continue
        refusal = agent.blockers()
        if not refusal:
            return memory
        agent._add_user("\n".join(refusal), "user")  # Extensions refuse to finish yet.
    raise BudgetExceeded("Agent reached max_turns")
