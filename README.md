# NFL Game Winner Prediction Model

A production-ready command-line project that trains a logistic regression model on
historical NFL schedules and predicts winners for upcoming games.

## What this project does

- Downloads historical and current NFL schedules using `nfl-data-py`.
- Builds team-strength features from games played **before** each matchup (to avoid leakage).
- Trains a `scikit-learn` logistic regression classifier.
- Prints model validation metrics.
- Predicts winners (with probabilities) for unplayed games in a target season/week.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py \
  --train-start 2015 \
  --train-end 2023 \
  --predict-season 2024 \
  --week 1 \
  --top 16
```

### CLI arguments

- `--train-start`: first season in training set (default: `2015`)
- `--train-end`: last season in training set (default: `2023`)
- `--predict-season`: season to generate predictions for (default: `2024`)
- `--week`: optional week filter (default: all weeks)
- `--top`: max number of rows to print (default: `16`)

## Feature engineering details

For each game, the model uses only information available before kickoff:

- Team identity: `home_team`, `away_team`
- Prior win percentage
- Prior average points scored/allowed
- Prior average point differential
- Recent-form win percentage (rolling last 5 games)
- Home-vs-away deltas for the metrics above
- Home-field indicator

## Output example

The script prints:

1. Validation accuracy + classification report on a holdout set.
2. Prediction table with:
   - away/home team
   - predicted winner
   - home and away win probabilities
   - model confidence

## Notes

- If no unplayed games exist for the requested season/week, the script exits gracefully.
- The model is deterministic via a fixed random seed (`42`).
