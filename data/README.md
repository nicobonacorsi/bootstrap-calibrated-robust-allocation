# Data

`run_experiment.py` downloads the daily U.S. 25 portfolios formed on Size and
Book-to-Market from the Kenneth R. French Data Library and caches them here.

You may instead pass a local date-indexed CSV of decimal returns:

```bash
python run_experiment.py --data data/my_returns.csv
```

Raw third-party data are intentionally not committed to the repository.
