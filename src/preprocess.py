"""
src/preprocess.py
Handles all data cleaning, encoding, scaling, and sequence creation for LSTM models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle
import os

# ─────────────────────────────────────────────
# SEQUENCE BUILDER FOR LSTM
# ─────────────────────────────────────────────

def create_sequences(data: np.ndarray, target: np.ndarray, seq_len: int = 10):
    """
    Converts flat time-series data into (X, y) sequences for LSTM.

    Args:
        data:    feature array shape (n_samples, n_features)
        target:  target array shape (n_samples,)
        seq_len: how many past matches to look at

    Returns:
        X: shape (n_sequences, seq_len, n_features)
        y: shape (n_sequences,)
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(target[i])
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
# BATTING PREPROCESSOR
# ─────────────────────────────────────────────

def preprocess_batting(csv_path: str = "data/raw/batting_stats.csv", seq_len: int = 10):
    df = pd.read_csv(csv_path)

    # Encode categoricals
    le_opp = LabelEncoder()
    le_venue = LabelEncoder()
    df["opposition_enc"] = le_opp.fit_transform(df["opposition_strength"])
    df["venue_enc"] = le_venue.fit_transform(df["venue"])

    feature_cols = ["runs", "balls_faced", "strike_rate", "fours", "sixes",
                    "opposition_enc", "venue_enc", "is_knockout", "form_index"]

    scaler = MinMaxScaler()
    all_X, all_y_runs, all_y_sr = [], [], []

    for player in df["player_name"].unique():
        pdata = df[df["player_name"] == player].sort_values("match_number").reset_index(drop=True)

        if len(pdata) < seq_len + 1:
            continue

        features = scaler.fit_transform(pdata[feature_cols].values)
        target_runs = pdata["runs"].values
        target_sr = pdata["strike_rate"].values

        X_r, y_r = create_sequences(features, target_runs, seq_len)
        X_s, y_s = create_sequences(features, target_sr, seq_len)

        all_X.append(X_r)
        all_y_runs.append(y_r)
        all_y_sr.append(y_s)

    X = np.concatenate(all_X, axis=0)
    y_runs = np.concatenate(all_y_runs, axis=0)
    y_sr = np.concatenate(all_y_sr, axis=0)

    # Save scaler and encoders
    os.makedirs("models", exist_ok=True)
    pickle.dump(scaler, open("models/batting_scaler.pkl", "wb"))
    pickle.dump(le_opp, open("models/batting_le_opp.pkl", "wb"))
    pickle.dump(le_venue, open("models/batting_le_venue.pkl", "wb"))

    print(f"Batting sequences: X={X.shape}, y_runs={y_runs.shape}, y_sr={y_sr.shape}")
    return X, y_runs, y_sr, feature_cols


# ─────────────────────────────────────────────
# BOWLING PREPROCESSOR
# ─────────────────────────────────────────────

def preprocess_bowling(csv_path: str = "data/raw/bowling_stats.csv", seq_len: int = 10):
    df = pd.read_csv(csv_path)

    le_opp = LabelEncoder()
    le_venue = LabelEncoder()
    df["opposition_enc"] = le_opp.fit_transform(df["opposition_strength"])
    df["venue_enc"] = le_venue.fit_transform(df["venue"])

    feature_cols = ["overs_bowled", "wickets", "runs_conceded", "economy_rate",
                    "dot_balls", "opposition_enc", "venue_enc", "is_knockout", "form_index"]

    scaler = MinMaxScaler()
    all_X, all_y_wk, all_y_eco = [], [], []

    for player in df["player_name"].unique():
        pdata = df[df["player_name"] == player].sort_values("match_number").reset_index(drop=True)

        if len(pdata) < seq_len + 1:
            continue

        features = scaler.fit_transform(pdata[feature_cols].values)
        target_wk = pdata["wickets"].values
        target_eco = pdata["economy_rate"].values

        X_w, y_w = create_sequences(features, target_wk, seq_len)
        X_e, y_e = create_sequences(features, target_eco, seq_len)

        all_X.append(X_w)
        all_y_wk.append(y_w)
        all_y_eco.append(y_e)

    X = np.concatenate(all_X, axis=0)
    y_wk = np.concatenate(all_y_wk, axis=0)
    y_eco = np.concatenate(all_y_eco, axis=0)

    pickle.dump(scaler, open("models/bowling_scaler.pkl", "wb"))
    pickle.dump(le_opp, open("models/bowling_le_opp.pkl", "wb"))
    pickle.dump(le_venue, open("models/bowling_le_venue.pkl", "wb"))

    print(f"Bowling sequences: X={X.shape}, y_wk={y_wk.shape}, y_eco={y_eco.shape}")
    return X, y_wk, y_eco, feature_cols


# ─────────────────────────────────────────────
# MATCH OUTCOME PREPROCESSOR
# ─────────────────────────────────────────────

def preprocess_match(csv_path: str = "data/raw/match_results.csv"):
    df = pd.read_csv(csv_path)

    le_team = LabelEncoder()
    le_venue = LabelEncoder()
    le_format = LabelEncoder()
    le_toss = LabelEncoder()
    le_decision = LabelEncoder()

    all_teams = pd.concat([df["team_a"], df["team_b"], df["toss_winner"]])
    le_team.fit(all_teams)

    df["team_a_enc"] = le_team.transform(df["team_a"])
    df["team_b_enc"] = le_team.transform(df["team_b"])
    df["toss_winner_enc"] = le_team.transform(df["toss_winner"])
    df["venue_enc"] = le_venue.fit_transform(df["venue"])
    df["format_enc"] = le_format.fit_transform(df["format"])
    df["toss_decision_enc"] = le_decision.fit_transform(df["toss_decision"])

    feature_cols = [
    "team_a_enc", "team_b_enc", "toss_winner_enc", "venue_enc",
    "format_enc", "toss_decision_enc",
    "team_a_score", "team_b_score",
    "team_a_wickets", "team_b_wickets",
    "team_a_run_rate", "team_b_run_rate",
    "overs_completed", "overs_remaining", "balls_remaining",
    "required_run_rate", "current_run_rate", "run_rate_diff",
    "wickets_remaining", "balls_per_wicket"
    ]

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    y = df["result"].values

    pickle.dump(scaler, open("models/match_scaler.pkl", "wb"))
    pickle.dump(le_team, open("models/match_le_team.pkl", "wb"))
    pickle.dump(le_venue, open("models/match_le_venue.pkl", "wb"))
    pickle.dump(le_format, open("models/match_le_format.pkl", "wb"))
    pickle.dump(le_decision, open("models/match_le_decision.pkl", "wb"))

    print(f"Match data: X={X.shape}, y={y.shape}")
    return X, y, feature_cols
