# When Can You Ship on Evals Alone? Trial-Level Surrogacy for Offline Evaluation of LLM Systems

Simulation code and empirical analysis for the paper.

## Reproducing the results

The simulation notebook needs only numpy, scipy, and matplotlib. The Upworthy notebook additionally needs the exploratory packages file from the Upworthy Research Archive (https://osf.io/jd64p), placed alongside the notebooks as `upworthy-archive-exploratory-packages.csv`.

```
pip install numpy scipy matplotlib pandas jupyter
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 simulations.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 upworthy.ipynb
```

Both notebooks run from fixed seeds and reproduce every number in the paper. The simulation notebook takes about four minutes.

## Files

- `simulations.ipynb` — all simulations, closed forms, and figures
- `upworthy.ipynb` — Upworthy illustration (Section 7.2)
- `judge_cache.json` — cached GPT-4o-mini headline ratings
- `build_notebooks.py` — generates both notebooks from source
