# Configuration files

| File | What it fixes |
| --- | --- |
| `stress.yaml` | The stress scenario (the decided parameter set). Pressure levels override `h0` and `accept_decay`. |
| `calibrated.yaml` | The calibrated scenario, below. |
| `small_exact.yaml` | The exact small instance solved by `hiring/exact_dp.py`. |
| `seeds.yaml` | Seed pools: development, tuning and evaluation episodes; tuning and final training seeds. |
| `ppo_small.yaml`, `ppo_stress.yaml` | MaskablePPO starting hyperparameters (not yet tuned). |

## Calibrated scenario: values and sources

The links are in `context/Hiring race RL research plan.md` ("Two parameter sets") and
`notes/evaluation_rigor_and_calibration.md`. "Secondary" means the figure was read in a summary,
not in the original paper; it must be opened before it is cited in the report.

| Parameter | Stress | Calibrated | Basis | Strength |
| --- | --- | --- | --- | --- |
| Horizon, roles, arrivals, interviews per day | 60, 3, 1.5, 3 | same | Time to fill 60 to 61 days at large employers (SHRM 2025); an arrival is a screened-in candidate | Supported |
| Acceptance decay per day | 0.03 | 0.003 | Acceptance of 76 to 84% (Ashby; Gem 2025) with offers weeks after application implies 0.002 to 0.0035 | Our inference; no source gives acceptance by day |
| Base acceptance | 0.90 | 0.80 | Notes' range is 0.85 to 0.90, but at 0.85 realised acceptance under the best tuned rule was 84.6%, outside the 76 to 84% benchmark. 0.80 aims at the middle of the benchmark | Our choice, below the notes' range, set to match the realised-acceptance benchmark (Nov 14, replacing 0.85) |
| Poaching level h0 | 0 to 0.10 | 0.02 | Stated disengagement of 12% at one week and 36% at one month implies 0.012 to 0.017 (Cronofy 2024) | Weak |
| Interview noise sd | 1.0 | 2.2 | Structured-interview validity r = .42 gives 2.16, from r = 1 / sqrt(1 + sd^2) | Secondary |
| Resume noise sd | 1.5 | 2.4 | Biodata validity r = .38 gives 2.43 | Secondary; our choice within "2.4 or more" (decided Nov 14) |
| Quality slope 0.6, exponential decay form, interview cost 0.05 | as decided | same | None found | Stated as a limitation |

Check before freezing (`results/g2/calibrated_check.txt`): realised acceptance under the best
tuned rule on the tuning episodes against 76 to 84%, share of roles filled, interviews per hire.
