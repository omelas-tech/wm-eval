| protocol | model | n | success [95% CI] | random | expert oracle | normalized | p vs random |
|---|---|---:|---|---|---|---:|---:|
| `tworoom/k50-b100` <sub>656334fa8569</sub> | `hf:quentinll/lewm-tworooms` | 200 | 61% [54–67] | 12% [9–18] | 100% [98–100] | 0.55 | 4.1e-27 |
| `tworoom/native-b100` <sub>7404395fb13b</sub> | `hf:quentinll/lewm-tworooms` | 200 | 34% [28–41] | 8% [5–13] | 100% [98–100] | 0.28 | 2.3e-13 |
| `tworoom/repo-k25` <sub>f1dbc2edc8c9</sub> | `hf:quentinll/lewm-tworooms` | 200 | 82% [77–87] | 24% [18–30] | 100% [98–100] | 0.77 | 1.8e-34 |
| `tworoom/repo-k25-hist3` <sub>3a333d9f3f10</sub> | `hf:quentinll/lewm-tworooms` | 200 | 83% [77–88] | 24% [18–30] | 100% [98–100] | 0.78 | 9.2e-35 |

Paired comparisons (same model, same episodes, different protocol; exact McNemar):

| model | protocol A | protocol B | A | B | only A | only B | p |
|---|---|---|---:|---:|---:|---:|---:|
| `hf:quentinll/lewm-tworooms` | `tworoom/repo-k25` | `tworoom/repo-k25-hist3` | 82% | 83% | 0 | 1 | 1 |
