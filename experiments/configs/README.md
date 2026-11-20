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
`notes/evaluation_rigor_and_calibration.md`. Every figure in the "Basis" column was read in the
source itself on Nov 20 (table "Sources opened" below); the BibTeX entries are in `paper/references.bib`.

| Parameter | Stress | Calibrated | Basis | Strength |
| --- | --- | --- | --- | --- |
| Horizon, roles, arrivals, interviews per day | 60, 3, 1.5, 3 | same | Time to fill 60 to 61 days at large employers (SHRM 2025); an arrival is a screened-in candidate | Supported |
| Acceptance decay per day | 0.03 | 0.003 | Acceptance of 76 to 84% (Ashby; Gem 2025) with offers weeks after application implies 0.002 to 0.0035 | Our inference; no source gives acceptance by day |
| Base acceptance | 0.90 | 0.80 | Notes' range is 0.85 to 0.90, but at 0.85 realised acceptance under the best tuned rule was 84.6%, outside the 76 to 84% benchmark. 0.80 aims at the middle of the benchmark | Our choice, below the notes' range, set to match the realised-acceptance benchmark (Nov 14, replacing 0.85) |
| Poaching level h0 | 0 to 0.10 | 0.02 | Stated disengagement of 12% at one week and 36% at one month implies 0.012 to 0.017 (Cronofy 2024) | Weak: figures confirmed, but they are stated intentions about scheduling delay, and the hazard is our inference |
| Interview noise sd | 1.0 | 2.2 | Structured-interview validity r = .42 (Sackett et al. 2023, reporting their 2022 estimates) gives 2.16, from r = 1 / sqrt(1 + sd^2) | Figure confirmed; the mapping to a noise sd is ours |
| Resume noise sd | 1.5 | 2.4 | Validity of empirically keyed biodata r = .38 (same source; rationally keyed biodata is .22) gives 2.43 | Figure confirmed; biodata as a stand-in for a resume and the value within "2.4 or more" are our choices (decided Nov 14) |
| Quality slope 0.6, exponential decay form, interview cost 0.05 | as decided | same | None found | Stated as a limitation |

### Sources opened (Nov 20, 2026)

| Source | What was read there | How it was opened |
| --- | --- | --- |
| SHRM 2025, "2025 Recruiting Executives Benchmarking" (data brief) | Median time to fill at organisations with 5,000 or more employees: 61 days nonexecutive, 60 days executive. 2,371 members responded, Jan 9 to Mar 3, 2025, unweighted | PDF downloaded, text searched |
| Gem 2025, "2025 Recruiting Benchmarks Report" | Offer-to-hire 81% (2021) and 84% (2024); by department in 2024, lowest Engineering 76%, highest Customer Service/Support 90%. Interviews per hire 20 in 2024 against 14 in 2021. Over 140 million applicants, 1.3 million hires, Jan 2021 to Dec 2024 | PDF downloaded, text searched |
| Ashby, "Offer Acceptance Rates" (Talent Trends Report) | Offer acceptance 77% (2021), 76% (2022), 81% (2023), 78% over the period; 73% technical, 84% business roles. 230K offers, Jan 2021 to Mar 2024 | Web page fetched, text searched |
| Cronofy, "Candidate Expectations Report 2024" | 36% of candidates would disengage at the one-month mark and 12% at one week. 12,000 candidates, seven countries | Web page fetched, text searched |
| Sackett, Zhang, Berry and Lievens (2023), Industrial and Organizational Psychology 16, 283 to 300 | Mean validity .42 for structured interviews and .38 for empirically keyed biodata (their 2022 estimates, Table 1 and text) | PDF downloaded, text searched |

Not opened, so not cited: Sackett et al. (2022, Journal of Applied Psychology), the original of the
validity estimates; the Gem 2026 summary; the Ashby recruiting-operations page. No figure from them
is used. The "76 to 84%" acceptance benchmark is the span of the overall figures above (Ashby by
year 76 to 81%, Gem 81 to 84%); Ashby's 73% for technical roles lies below it.

Check before freezing (`results/g2/calibrated_check.txt`): realised acceptance under the best
tuned rule on the tuning episodes against 76 to 84%, share of roles filled, interviews per hire.
