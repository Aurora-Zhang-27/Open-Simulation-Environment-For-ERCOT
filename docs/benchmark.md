# Scoring an agent on RTCB-Gym

Any agent (a rule, an optimiser, a reinforcement-learning policy, a
language model) is scored the same way: on a named **task**, through one
function, against an exact bound, into a **scorecard**.

```bash
python -m rtcb_gym.evaluate --list
python -m rtcb_gym.evaluate --task rt-energy --agent examples/agents/threshold.py --source synthetic
```

`--source synthetic` needs no market data and no API key, so you can
develop an agent on a fresh clone. Synthetic scores only show that the
agent runs; results need `--source real` (see the README for the data).

## Tasks

A task fixes everything except the agent: market setting, battery,
episode length, splits and bound.

| Task | Markets | Action | Episode |
|---|---|---|---|
| `rt-energy` | real-time energy, 15 min | power | one day, starts empty |
| `rt-energy-offer` | the same | offer curve applied to the interval's own price | one day, starts empty |
| `da-rt-energy` | day-ahead + real-time energy | hourly bid, then power | one day, starts empty |
| `da-rt-energy-7day` | the same | the same | 7 days, half full at both ends, SoC carries over midnight |
| `da-rt-energy-ecrs-5min` | day-ahead + real time, energy and ECRS, 5-min dispatch, 15-min settlement | hourly bid for both, then power and reserve | one day, half full at both ends |

Battery, dates and hub come from the config (`configs/default.yaml`:
100 MW / 200 MWh at HB_WEST). The five-minute task needs the SCED cache
(`scripts/download_5min.py`); on real data it currently covers July to
October 2026 plus two winter weeks, so its training split is short.

## Writing an agent

An agent is a `Policy` built by a function that takes the task spec:

```python
import numpy as np
from rtcb_gym.baselines import Policy

class Threshold(Policy):
    name = "threshold"

    def __init__(self, spec):
        self.spec = spec
        prices = spec.train["rt"]["price_usd_per_mwh"]       # training split only
        self.low, self.high = np.quantile(prices, [0.3, 0.7])

    def act(self, obs, info):
        last = obs[self.spec.obs_layout["prices"]][-1]       # newest settled price
        if last <= self.low:
            return self.spec.action(-1.0)                    # charge
        if last >= self.high:
            return self.spec.action(1.0)                     # discharge
        return self.spec.action(0.0)

def make_agent(spec):
    return Threshold(spec)
```

| Method | When | Returns |
|---|---|---|
| `__init__(spec)` | once | |
| `day_ahead(context)` | once per operating day, tasks with a day-ahead market | hourly MW schedule, a `DayAheadBid`, or `None` |
| `reset(obs, info)` | start of each episode | |
| `act(obs, info)` | every interval | the action |

**What the agent gets.** The `TaskSpec`: battery, step length, the
layouts of observation and action, `spec.describe()` (the rules in plain
language) and `spec.train` (the task's price series for the training
split). Then observations. It never gets the environment, which holds the
prices it is being tested on.

**Actions.** `spec.action(energy, ecrs)` builds the right action for any
task from a power share in [-1, 1] and a reserve share in [0, 1]. Agents
that want the offer curve or other details write the array themselves
using `spec.action_layout`.

**Training a learning agent.** `rtcb_gym.benchmark.make_env(task,
split="train")` returns the task's Gymnasium environment on the training
split; wrap it in `rtcb_gym.ScaledObservation` for bounded inputs. Train
there, then return a `Policy` that calls your model.

## Language models

`rtcb_gym.text.TextAgent` turns each observation into a prompt (the rules,
then the state as JSON), passes it to a function you provide and parses
the JSON reply:

```python
from rtcb_gym.text import TextAgent

def complete(prompt: str) -> str:
    return call_your_model(prompt)          # reply e.g. {"energy": -0.5}

def make_agent(spec):
    return TextAgent(spec, complete, decide_every=4)   # one call an hour at 15-minute steps
```

The library makes no network calls and names no provider.
`examples/agents/text_scripted.py` runs the whole path offline with a
stand-in for the model. The model sees what a numeric agent sees: past
prices, the state of charge, the published day-ahead prices and awards.
One day is 96 intervals (288 at five minutes); `decide_every` sets how
many of them share a decision.

## The scorecard

`evaluate` returns, and `--out` writes, a JSON scorecard:

| Field | Meaning |
|---|---|
| `revenue_usd`, `revenue_ci95_usd` | net revenue and its interval over days |
| `bound_revenue_usd` | perfect foresight in the same task on the same days |
| `capture_rate`, `capture_ci95` | revenue / bound, with a block-bootstrap interval |
| `invalid_actions`, `invalid_day_ahead_bids` | times the agent raised, or returned something that is not an action; the battery idled |
| `clipped_actions` | requests the battery could not deliver; the feasible part was executed |
| `agent_seconds_per_step` | time spent inside the agent |
| `task_sha256`, `data`, `version` | what exactly was run: task definition, data hashes or synthetic seed, package version |
| `partial` | true when `--limit-days` was used |

`python -m rtcb_gym.evaluate --leaderboard cards/*.json` prints a table.

## Rules for a fair score

- Use the validation split while developing. Score the test split once,
  by name (`--split test`), when the agent is final.
- An agent may read `spec.train` and its observations, nothing else. The
  benchmark cannot stop code that opens the data files directly; a score
  from such an agent is not a score.
- Report the interval with the capture rate. With about 55 days per
  split, differences of a few points between agents are not established.
- Say which hub, battery and package version the scorecard is from; they
  are all in the file.

## Limits

- Historical replay: the agent is a price-taker. Nothing here tests an
  agent's effect on the market, or other agents.
- One battery at one trading hub.
- ECRS is a reduced model (see [ecrs_gate.md](ecrs_gate.md)).
- The reference agents are simple. The best of them captures about a
  third of the bound, so the leaderboard is a starting line.
