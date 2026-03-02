"""NFL game winner prediction CLI.

This script trains a logistic regression model on historical NFL schedule data and
predicts winners for upcoming games.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    import nfl_data_py as nfl
except ImportError:  # pragma: no cover - import guard
    nfl = None


FEATURE_COLUMNS = [
    "home_team",
    "away_team",
    "home_win_pct",
    "away_win_pct",
    "win_pct_diff",
    "home_avg_points_for",
    "away_avg_points_for",
    "points_for_diff",
    "home_avg_points_against",
    "away_avg_points_against",
    "points_against_diff",
    "home_avg_point_diff",
    "away_avg_point_diff",
    "point_diff_diff",
    "home_recent_win_pct",
    "away_recent_win_pct",
    "recent_form_diff",
    "home_field",
]


@dataclass
class ModelArtifacts:
    model: Pipeline
    feature_columns: list[str]
    test_accuracy: float




def _print_metric_interpretation(y_test: pd.Series, predictions: np.ndarray, accuracy: float) -> None:
    """Print a short, human-readable explanation of evaluation metrics."""
    test_size = len(y_test)
    home_rate = float(y_test.mean()) if test_size else 0.0
    predicted_home_rate = float(np.mean(predictions)) if test_size else 0.0
    baseline = max(home_rate, 1 - home_rate)

    print("Metric interpretation")
    print("=" * 60)
    print(f"Holdout games evaluated: {test_size}")
    print(f"True home-win rate in holdout: {home_rate:.1%}")
    print(f"Model predicted home wins: {predicted_home_rate:.1%} of games")
    print(f"Accuracy: {accuracy:.1%} (baseline if always pick majority class: {baseline:.1%})")
    print(
        "Precision answers 'when the model predicts this class, how often is it right?', "
        "while recall answers 'of all real games in this class, how many did the model find?'."
    )


def _find_available_unplayed_windows(schedule: pd.DataFrame) -> tuple[list[int], list[tuple[int, int]]]:
    if not {"home_score", "away_score"}.issubset(schedule.columns):
        return sorted(schedule["season"].dropna().astype(int).unique().tolist()), []

    unplayed = schedule[schedule["home_score"].isna() | schedule["away_score"].isna()]
    seasons = sorted(unplayed["season"].dropna().astype(int).unique().tolist())
    season_week_pairs = (
        unplayed[["season", "week"]]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values(["season", "week"])
    )
    pairs = list(season_week_pairs.itertuples(index=False, name=None))
    return seasons, pairs

class NFLGamePredictor:
    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    @staticmethod
    def _normalize_schedule_columns(schedule: pd.DataFrame) -> pd.DataFrame:
        """Normalize schedule columns across nfl_data_py versions."""
        renamed = schedule.copy()
        aliases = {
            "gameday": ["game_date", "date"],
            "home_score": ["score_home", "home_points"],
            "away_score": ["score_away", "away_points"],
            "home_team": ["team_home"],
            "away_team": ["team_away"],
        }
        for canonical, candidates in aliases.items():
            if canonical in renamed.columns:
                continue
            for candidate in candidates:
                if candidate in renamed.columns:
                    renamed[canonical] = renamed[candidate]
                    break

        required = ["season", "week", "home_team", "away_team", "gameday"]
        missing = [column for column in required if column not in renamed.columns]
        if missing:
            raise ValueError(f"Missing required schedule columns: {missing}")

        renamed["gameday"] = pd.to_datetime(renamed["gameday"], errors="coerce")
        renamed = renamed.sort_values(["season", "week", "gameday"]).reset_index(drop=True)
        return renamed

    def load_schedule(self, seasons: Iterable[int]) -> pd.DataFrame:
        if nfl is None:
            raise ImportError(
                "nfl_data_py is required. Install dependencies with: pip install -r requirements.txt"
            )

        season_list = sorted(set(int(year) for year in seasons))
        schedule = nfl.import_schedules(season_list)
        return self._normalize_schedule_columns(schedule)

    @staticmethod
    def _build_team_game_rows(completed_games: pd.DataFrame) -> pd.DataFrame:
        home_rows = completed_games[["season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"]].copy()
        home_rows.rename(
            columns={
                "home_team": "team",
                "away_team": "opponent",
                "home_score": "points_for",
                "away_score": "points_against",
            },
            inplace=True,
        )

        away_rows = completed_games[["season", "week", "gameday", "home_team", "away_team", "home_score", "away_score"]].copy()
        away_rows.rename(
            columns={
                "away_team": "team",
                "home_team": "opponent",
                "away_score": "points_for",
                "home_score": "points_against",
            },
            inplace=True,
        )

        team_games = pd.concat([home_rows, away_rows], ignore_index=True)
        team_games["point_diff"] = team_games["points_for"] - team_games["points_against"]
        team_games["win"] = (team_games["points_for"] > team_games["points_against"]).astype(int)
        team_games = team_games.sort_values(["team", "season", "week", "gameday"]).reset_index(drop=True)

        grouped = team_games.groupby("team", group_keys=False)
        shifted_win = grouped["win"].shift(1)
        shifted_pf = grouped["points_for"].shift(1)
        shifted_pa = grouped["points_against"].shift(1)
        shifted_pd = grouped["point_diff"].shift(1)

        team_games["games_played_before"] = grouped.cumcount()
        gp = team_games["games_played_before"].replace(0, np.nan)

        team_games["win_pct_before"] = grouped["win"].cumsum().shift(1) / gp
        team_games["avg_points_for_before"] = grouped["points_for"].cumsum().shift(1) / gp
        team_games["avg_points_against_before"] = grouped["points_against"].cumsum().shift(1) / gp
        team_games["avg_point_diff_before"] = grouped["point_diff"].cumsum().shift(1) / gp

        team_games["recent_win_pct_before"] = shifted_win.groupby(team_games["team"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        team_games["recent_points_for_before"] = shifted_pf.groupby(team_games["team"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        team_games["recent_points_against_before"] = shifted_pa.groupby(team_games["team"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
        team_games["recent_point_diff_before"] = shifted_pd.groupby(team_games["team"]).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

        return team_games

    def _create_feature_frame(self, schedule: pd.DataFrame, completed_only: bool = False) -> pd.DataFrame:
        score_cols_present = {"home_score", "away_score"}.issubset(schedule.columns)

        if score_cols_present:
            completed_mask = schedule["home_score"].notna() & schedule["away_score"].notna()
            completed_games = schedule.loc[completed_mask].copy()
        else:
            completed_games = schedule.iloc[0:0].copy()

        team_games = self._build_team_game_rows(completed_games)

        home_stats = team_games[[
            "season",
            "week",
            "gameday",
            "team",
            "win_pct_before",
            "avg_points_for_before",
            "avg_points_against_before",
            "avg_point_diff_before",
            "recent_win_pct_before",
        ]].copy()
        away_stats = home_stats.copy()

        home_stats.rename(
            columns={
                "team": "home_team",
                "win_pct_before": "home_win_pct",
                "avg_points_for_before": "home_avg_points_for",
                "avg_points_against_before": "home_avg_points_against",
                "avg_point_diff_before": "home_avg_point_diff",
                "recent_win_pct_before": "home_recent_win_pct",
            },
            inplace=True,
        )
        away_stats.rename(
            columns={
                "team": "away_team",
                "win_pct_before": "away_win_pct",
                "avg_points_for_before": "away_avg_points_for",
                "avg_points_against_before": "away_avg_points_against",
                "avg_point_diff_before": "away_avg_point_diff",
                "recent_win_pct_before": "away_recent_win_pct",
            },
            inplace=True,
        )

        merged = schedule.merge(home_stats, on=["season", "week", "gameday", "home_team"], how="left")
        merged = merged.merge(away_stats, on=["season", "week", "gameday", "away_team"], how="left")

        merged["win_pct_diff"] = merged["home_win_pct"] - merged["away_win_pct"]
        merged["points_for_diff"] = merged["home_avg_points_for"] - merged["away_avg_points_for"]
        merged["points_against_diff"] = merged["home_avg_points_against"] - merged["away_avg_points_against"]
        merged["point_diff_diff"] = merged["home_avg_point_diff"] - merged["away_avg_point_diff"]
        merged["recent_form_diff"] = merged["home_recent_win_pct"] - merged["away_recent_win_pct"]
        merged["home_field"] = 1

        if completed_only:
            merged = merged.loc[merged["home_score"].notna() & merged["away_score"].notna()].copy()
            merged["home_win"] = (merged["home_score"] > merged["away_score"]).astype(int)

        return merged

    def build_model(self) -> Pipeline:
        numeric_features = [col for col in FEATURE_COLUMNS if col not in {"home_team", "away_team"}]
        categorical_features = ["home_team", "away_team"]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    numeric_features,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("encoder", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                    categorical_features,
                ),
            ]
        )

        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", LogisticRegression(max_iter=2000, random_state=self.random_state)),
            ]
        )

    def train(self, schedule: pd.DataFrame, holdout_size: float = 0.2) -> ModelArtifacts:
        dataset = self._create_feature_frame(schedule, completed_only=True)
        if dataset.empty:
            raise ValueError("No completed games found. Cannot train model.")

        X = dataset[FEATURE_COLUMNS]
        y = dataset["home_win"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=holdout_size,
            random_state=self.random_state,
            stratify=y,
        )

        model = self.build_model()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        print("\nValidation report")
        print("=" * 60)
        print(f"Accuracy: {accuracy:.3f}")
        print(classification_report(y_test, predictions, target_names=["Away Win", "Home Win"]))
        _print_metric_interpretation(y_test, predictions, accuracy)

        return ModelArtifacts(model=model, feature_columns=FEATURE_COLUMNS, test_accuracy=accuracy)

    def predict_unplayed_games(
        self,
        model: Pipeline,
        schedule: pd.DataFrame,
        season: int,
        week: Optional[int] = None,
    ) -> pd.DataFrame:
        feature_frame = self._create_feature_frame(schedule, completed_only=False)

        mask = feature_frame["season"].eq(season)
        if week is not None:
            mask &= feature_frame["week"].eq(week)

        if {"home_score", "away_score"}.issubset(feature_frame.columns):
            unplayed_mask = feature_frame["home_score"].isna() | feature_frame["away_score"].isna()
        else:
            unplayed_mask = pd.Series(True, index=feature_frame.index)

        games_to_predict = feature_frame.loc[mask & unplayed_mask].copy()
        if games_to_predict.empty:
            return games_to_predict

        probs = model.predict_proba(games_to_predict[FEATURE_COLUMNS])[:, 1]
        games_to_predict["home_win_probability"] = probs
        games_to_predict["away_win_probability"] = 1 - probs
        games_to_predict["predicted_winner"] = np.where(
            games_to_predict["home_win_probability"] >= 0.5,
            games_to_predict["home_team"],
            games_to_predict["away_team"],
        )
        games_to_predict["confidence"] = np.maximum(
            games_to_predict["home_win_probability"], games_to_predict["away_win_probability"]
        )

        columns = [
            "season",
            "week",
            "gameday",
            "away_team",
            "home_team",
            "predicted_winner",
            "home_win_probability",
            "away_win_probability",
            "confidence",
        ]
        return games_to_predict[columns].sort_values(["week", "gameday", "home_team"])


def _prompt_int(prompt: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter a whole number.")
                continue

        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Please enter a value <= {maximum}.")
            continue
        return value


def _prompt_optional_int(prompt: str, default: Optional[int], minimum: int | None = None, maximum: int | None = None) -> Optional[int]:
    shown_default = "all weeks" if default is None else str(default)
    while True:
        raw = input(f"{prompt} [{shown_default}]: ").strip().lower()
        if raw in {"", "all", "none"}:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number, or press Enter for all weeks.")
            continue

        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Please enter a value <= {maximum}.")
            continue
        return value


def collect_user_inputs() -> tuple[int, int, int, Optional[int], int]:
    print("Enter model settings (press Enter to use defaults):")
    train_start = _prompt_int("Train start season", 2015, minimum=1999)
    train_end = _prompt_int("Train end season", 2023, minimum=train_start)
    predict_season = _prompt_int("Predict season", 2024, minimum=train_end)
    week = _prompt_optional_int("Predict week (or Enter for all)", None, minimum=1, maximum=22)
    top = _prompt_int("How many predicted games to show", 16, minimum=1, maximum=272)
    return train_start, train_end, predict_season, week, top


def main() -> int:
    train_start, train_end, predict_season, week, top = collect_user_inputs()

    predictor = NFLGamePredictor(random_state=42)
    all_seasons = list(range(train_start, predict_season + 1))

    print(f"Loading schedules for seasons: {all_seasons[0]}-{all_seasons[-1]}...")
    schedule = predictor.load_schedule(all_seasons)

    train_mask = (schedule["season"] >= train_start) & (schedule["season"] <= train_end)
    train_schedule = schedule.loc[train_mask].copy()
    print(f"Training games available: {len(train_schedule)}")

    artifacts = predictor.train(train_schedule)

    predictions = predictor.predict_unplayed_games(
        model=artifacts.model,
        schedule=schedule,
        season=predict_season,
        week=week,
    )

    if predictions.empty:
        print("No unplayed games found for requested season/week.")
        available_seasons, available_pairs = _find_available_unplayed_windows(schedule)
        if available_seasons:
            season_list = ", ".join(str(season) for season in available_seasons)
            print(f"Unplayed games exist for season(s): {season_list}")
        if available_pairs:
            preview_pairs = ", ".join(f"{season}-W{week}" for season, week in available_pairs[:12])
            suffix = " ..." if len(available_pairs) > 12 else ""
            print(f"Available season/week values include: {preview_pairs}{suffix}")
        else:
            print("All imported games appear completed. Try a future --predict-season.")
        return 0

    print("\nPredictions")
    print("=" * 60)
    preview = predictions.head(top).copy()
    preview["home_win_probability"] = (preview["home_win_probability"] * 100).round(1)
    preview["away_win_probability"] = (preview["away_win_probability"] * 100).round(1)
    preview["confidence"] = (preview["confidence"] * 100).round(1)
    print(preview.to_string(index=False))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - top-level error display
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
