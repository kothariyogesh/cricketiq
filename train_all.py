import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  CRICKET PERFORMANCE PREDICTOR — FULL TRAINING PIPELINE")
print("=" * 60)

print("\n[1/4] Generating datasets...")
from generate_data import generate_batting_data, generate_bowling_data, generate_match_data

os.makedirs("data/raw", exist_ok=True)
generate_batting_data().to_csv("data/raw/batting_stats.csv", index=False)
generate_bowling_data().to_csv("data/raw/bowling_stats.csv", index=False)
generate_match_data().to_csv("data/raw/match_results.csv", index=False)
print("  Done.")

print("\n[2/4] Preprocessing data...")
from src.preprocess import preprocess_batting, preprocess_bowling, preprocess_match

X_bat, y_runs, y_sr, _ = preprocess_batting()
X_bowl, y_wk, y_eco, _ = preprocess_bowling()
X_match, y_match, _ = preprocess_match()
print("  Done.")

print("\n[3/4] Training models...")
from src.model_batting import train_batting_model
from src.model_bowling import train_bowling_model
from src.model_match import train_match_model

train_batting_model(X_bat, y_runs, y_sr)
train_bowling_model(X_bowl, y_wk, y_eco)
train_match_model(X_match, y_match)

print("\n[4/4] All models trained and saved!")
print("\nModels saved in models/")
print("Plots saved in plots/")
print("\nNow run: streamlit run app.py")
print("=" * 60)