"""How a language model plugs in, with a stand-in for the model.

    python -m rtcb_gym.evaluate --task da-rt-energy --agent examples/agents/text_scripted.py --source synthetic

``complete`` receives the prompt a model would get (the task rules, then
the current state as JSON) and must return JSON. Here it is a few lines of
Python reading that same JSON, so the example runs offline and costs
nothing. Replace its body with a call to your model.
"""

import json

from rtcb_gym.text import TextAgent


def complete(prompt: str) -> str:
    state = json.loads(prompt.split("Current state:\n", 1)[1].split("\n\n", 1)[0])
    if "hours_to_bid" in state:  # the day-ahead question: no position
        return json.dumps({"energy_mw": [0.0] * state["hours_to_bid"]})
    recent = state["recent_prices_usd_per_mwh"]
    median = state["price_history"]["median"]
    if recent[-1] < 0.7 * median:
        return 'I will charge. {"energy": -1.0}'  # prose around the JSON is tolerated
    if recent[-1] > 1.3 * median:
        return '{"energy": 1.0}'
    return '{"energy": 0.0}'


def make_agent(spec):
    return TextAgent(spec, complete, decide_every=4, name="example: scripted text agent")
