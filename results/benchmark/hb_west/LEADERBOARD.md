# RTCB-Gym benchmark: HB_WEST

Reference and example agents on every task. Capture rate = revenue / perfect foresight in the same task on the same days. These agents are simple on purpose; see docs/benchmark.md for how to add yours.

| Task | Split | Data | Agent | Days | Revenue (USD) | Capture rate (95% interval) | Invalid / clipped actions | Seconds per step |
|---|---|---|---|---:|---:|---:|---:|---:|
| rt-energy | test | real | persistence MPC | 59 | 435,338 | 46.9% (33.1 to 58.9) | 0 / 0 | 0.0052 |
| rt-energy | test | real | example: scripted text agent | 59 | 324,756 | 35.0% (19.3 to 50.9) | 0 / 2142 | 0.0000 |
| rt-energy | test | real | example: threshold | 59 | 96,692 | 10.4% (2.7 to 21.6) | 0 / 3301 | 0.0000 |
| rt-energy | test | real | idle | 59 | 0 | 0.0% (0.0 to 0.0) | 0 / 0 | 0.0000 |
| rt-energy | validation | real | persistence MPC | 53 | 267,207 | 49.6% (29.7 to 68.1) | 0 / 0 | 0.0050 |
| rt-energy | validation | real | example: scripted text agent | 53 | 162,401 | 30.2% (17.6 to 46.2) | 0 / 1482 | 0.0000 |
| rt-energy | validation | real | example: threshold | 53 | 65,042 | 12.1% (4.5 to 24.6) | 0 / 1560 | 0.0000 |
| rt-energy | validation | real | idle | 53 | 0 | 0.0% (0.0 to 0.0) | 0 / 0 | 0.0000 |
| rt-energy-offer | test | real | persistence MPC | 59 | 435,338 | 46.9% (33.1 to 58.9) | 0 / 0 | 0.0050 |
| rt-energy-offer | test | real | example: threshold | 59 | 96,692 | 10.4% (2.7 to 21.6) | 0 / 3301 | 0.0000 |
| rt-energy-offer | validation | real | persistence MPC | 53 | 267,207 | 49.6% (29.7 to 68.1) | 0 / 0 | 0.0050 |
| rt-energy-offer | validation | real | example: threshold | 53 | 65,042 | 12.1% (4.5 to 24.6) | 0 / 1560 | 0.0000 |
| da-rt-energy | test | real | day-ahead schedule, delivered | 59 | 534,222 | 29.1% (22.9 to 37.0) | 0 / 0 | 0.0000 |
| da-rt-energy | test | real | persistence MPC | 59 | 435,338 | 23.7% (15.9 to 31.2) | 0 / 0 | 0.0050 |
| da-rt-energy | test | real | example: scripted text agent | 59 | 324,756 | 17.7% (9.6 to 26.5) | 0 / 2142 | 0.0000 |
| da-rt-energy | test | real | example: threshold | 59 | 96,692 | 5.3% (1.4 to 10.8) | 0 / 3301 | 0.0000 |
| da-rt-energy | validation | real | day-ahead schedule, delivered | 53 | 403,959 | 36.0% (28.2 to 45.5) | 0 / 0 | 0.0000 |
| da-rt-energy | validation | real | persistence MPC | 53 | 267,207 | 23.8% (13.8 to 34.1) | 0 / 0 | 0.0047 |
| da-rt-energy | validation | real | example: scripted text agent | 53 | 162,401 | 14.5% (8.5 to 21.7) | 0 / 1482 | 0.0000 |
| da-rt-energy | validation | real | example: threshold | 53 | 65,042 | 5.8% (2.1 to 11.9) | 0 / 1560 | 0.0000 |
| da-rt-energy-7day | test | real | day-ahead schedule, delivered | 56 | 447,502 | 28.6% (22.3 to 36.2) | 0 / 0 | 0.0000 |
| da-rt-energy-7day | test | real | persistence MPC | 56 | 361,327 | 23.1% (12.9 to 33.0) | 0 / 0 | 0.0103 |
| da-rt-energy-7day | validation | real | day-ahead schedule, delivered | 49 | 301,353 | 34.4% (25.6 to 42.6) | 0 / 0 | 0.0000 |
| da-rt-energy-7day | validation | real | persistence MPC | 49 | 246,581 | 28.2% (18.3 to 35.6) | 0 / 0 | 0.0105 |
| da-rt-energy-ecrs-5min | test | real | day-ahead schedule, delivered | 57 | 437,442 | 25.0% (19.3 to 32.6) | 0 / 0 | 0.0000 |
| da-rt-energy-ecrs-5min | test | real | example: threshold | 57 | 267,912 | 15.3% (8.6 to 22.0) | 0 / 5223 | 0.0000 |
| da-rt-energy-ecrs-5min | validation | real | day-ahead schedule, delivered | 51 | 326,652 | 30.5% (23.2 to 38.7) | 0 / 0 | 0.0000 |
| da-rt-energy-ecrs-5min | validation | real | example: threshold | 51 | 82,019 | 7.7% (-4.3 to 16.2) | 0 / 5966 | 0.0000 |

Data provided by [Grid Status](https://www.gridstatus.io).
