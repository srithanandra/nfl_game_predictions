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

Run the script and answer prompts:

```bash
python main.py
```

You will be prompted for:

- train start season (default `2015`)
- train end season (default `2023`)
- predict season (default `2024`)
- predict week or all weeks (default: all weeks)
- number of predictions to print (default `16`)

You can press **Enter** at each prompt to accept defaults.

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
