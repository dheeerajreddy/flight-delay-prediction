# Flight Arrival Delay Prediction

Predict how many minutes late a flight will **arrive**, using only information
available **before** it departs. Refactored from a single Colab notebook into a
config-driven, testable, reusable pipeline.

---

## Why this structure

| Requirement | How it's met |
|---|---|
| **Parameterisation** | Every path, hyper-parameter, and modelling choice lives in `config/config.yaml`. Source code reads config — it never hardcodes. |
| **Reusability** | Each stage (`load`, `clean`, `split`, `features`, `train`) is a pure module you can import and reuse independently. |
| **Scalability** | Weather is cached to parquet; airports come from a CSV lookup, not a hardcoded dict — add destinations without touching code. |
| **Correctness** | Temporal split + train-only imputation + target sanity-clipping fix the leakage bugs from the original notebook. |
| **Testability** | `tests/` covers the exact failure modes the notebook had. |
| **Reproducibility** | Single seed in config; artifacts (model, preprocessor, leaderboard) persisted to `artifacts/`. |

---

## Layout

```
flight-delay-prediction/
├── config/
│   ├── config.yaml         # all parameters
│   └── airports.csv        # IATA -> lat/lon lookup
├── data/
│   ├── raw/                # input CSV (git-ignored)
│   ├── interim/            # weather cache
│   └── processed/          # engineered dataset
├── src/
│   ├── config.py           # config loader
│   ├── data/               # load, clean, split
│   ├── features/           # temporal, weather, preprocessing
│   ├── models/             # registry, train/evaluate
│   └── pipeline.py         # wires the stages together
├── scripts/run_pipeline.py # CLI entry point
├── tests/                  # pytest
└── artifacts/              # saved models + reports
```

---

## Run it

```bash
pip install -r requirements.txt
# place the raw file at data/raw/flights.csv (or edit config)
python -m scripts.run_pipeline
pytest -q
```

Swap any behaviour without editing code — e.g. run a random-split baseline:

```bash
cp config/config.yaml config/random_baseline.yaml
# set split.strategy: random  in that file
python -m scripts.run_pipeline --config config/random_baseline.yaml
```

---

## Known limitations (inherited from the source, documented not hidden)

1. **Weather is observed, not forecast.** The archive API returns what actually
   happened at arrival hour; a production model needs the forecast known at
   scheduling time. Swap the endpoint in config when ready.
2. **`MINS_TO_SCHD_DEP_QTY` dominates feature importance.** Confirm it is truly
   knowable pre-departure before trusting it.
3. **Target distribution looked implausible** (mean ~58 min) in the source data.
   The pipeline now drops out-of-range delays, but the upstream timezone/parse
   logic should be verified against known flights.
```
