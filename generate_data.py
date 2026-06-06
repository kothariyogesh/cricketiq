import pandas as pd
import numpy as np
import os

np.random.seed(42)

NUM_PLAYERS = 50
MATCHES_PER_PLAYER = 60

PLAYER_NAMES = [
    "Rohit Sharma", "Virat Kohli", "KL Rahul", "Shubman Gill", "Rishabh Pant",
    "Hardik Pandya", "Ravindra Jadeja", "Axar Patel", "Washington Sundar", "Venkatesh Iyer",
    "Shreyas Iyer", "Suryakumar Yadav", "Ishan Kishan", "Ruturaj Gaikwad", "Sanju Samson",
    "Yashasvi Jaiswal", "Tilak Varma", "Rinku Singh", "Shikhar Dhawan", "Ajinkya Rahane",
    "Jasprit Bumrah", "Mohammed Shami", "Yuzvendra Chahal", "Kuldeep Yadav", "Deepak Chahar",
    "Bhuvneshwar Kumar", "Arshdeep Singh", "Mohammed Siraj", "Umesh Yadav", "Ravichandran Ashwin",
    "David Warner", "Steve Smith", "Pat Cummins", "Mitchell Starc", "Glenn Maxwell",
    "Jos Buttler", "Ben Stokes", "Jofra Archer", "Joe Root", "Jonny Bairstow",
    "Babar Azam", "Shaheen Afridi", "Mohammad Rizwan", "Shadab Khan", "Naseem Shah",
    "Quinton de Kock", "Kagiso Rabada", "Aiden Markram", "Lungi Ngidi", "David Miller",
]

ARCHETYPES = {
    "aggressive_batter": (55, 20, 145, 0.05, 8.5),
    "anchor_batter":     (45, 15, 118, 0.05, 8.0),
    "allrounder":        (35, 18, 130, 0.25, 7.5),
    "pace_bowler":       (10,  8,  90, 0.45, 7.0),
    "spin_bowler":       (12,  8,  85, 0.40, 6.5),
}

def assign_archetype(player_idx):
    if player_idx < 10:
        return "aggressive_batter"
    elif player_idx < 20:
        return "anchor_batter"
    elif player_idx < 30:
        return "allrounder"
    elif player_idx < 40:
        return "pace_bowler"
    else:
        return "spin_bowler"

def generate_batting_data():
    records = []
    for i, player in enumerate(PLAYER_NAMES):
        arch = assign_archetype(i)
        avg_r, std_r, avg_sr, _, _ = ARCHETYPES[arch]
        form_trend = np.linspace(-5, 5, MATCHES_PER_PLAYER) + np.random.randn(MATCHES_PER_PLAYER) * 3
        for match in range(MATCHES_PER_PLAYER):
            runs = max(0, int(np.random.normal(avg_r + form_trend[match], std_r)))
            balls = max(1, int(runs / (avg_sr / 100) + np.random.randint(-5, 10)))
            strike_rate = round((runs / balls) * 100, 2)
            fours = int(runs * np.random.uniform(0.05, 0.12))
            sixes = int(runs * np.random.uniform(0.02, 0.08))
            opposition_strength = np.random.choice(["weak", "medium", "strong"])
            venue = np.random.choice(["home", "away", "neutral"])
            is_knockout = int(np.random.random() < 0.2)
            records.append({
                "player_name": player,
                "archetype": arch,
                "match_number": match + 1,
                "runs": runs,
                "balls_faced": balls,
                "strike_rate": strike_rate,
                "fours": fours,
                "sixes": sixes,
                "opposition_strength": opposition_strength,
                "venue": venue,
                "is_knockout": is_knockout,
                "form_index": round(form_trend[match], 2)
            })
    return pd.DataFrame(records)

def generate_bowling_data():
    records = []
    for i, player in enumerate(PLAYER_NAMES):
        arch = assign_archetype(i)
        _, _, _, wicket_prob, avg_eco = ARCHETYPES[arch]
        form_trend = np.linspace(-0.5, 0.5, MATCHES_PER_PLAYER) + np.random.randn(MATCHES_PER_PLAYER) * 0.2
        for match in range(MATCHES_PER_PLAYER):
            overs = round(np.random.uniform(2, 4), 1)
            wickets = np.random.binomial(6, wicket_prob)
            economy = max(3.0, round(np.random.normal(avg_eco + form_trend[match], 1.2), 2))
            runs_given = int(overs * economy)
            dot_balls = int(overs * 6 * np.random.uniform(0.2, 0.45))
            opposition_strength = np.random.choice(["weak", "medium", "strong"])
            venue = np.random.choice(["home", "away", "neutral"])
            is_knockout = int(np.random.random() < 0.2)
            records.append({
                "player_name": player,
                "archetype": arch,
                "match_number": match + 1,
                "overs_bowled": overs,
                "wickets": wickets,
                "runs_conceded": runs_given,
                "economy_rate": economy,
                "dot_balls": dot_balls,
                "opposition_strength": opposition_strength,
                "venue": venue,
                "is_knockout": is_knockout,
                "form_index": round(form_trend[match], 2)
            })
    return pd.DataFrame(records)

def generate_match_data():
    records = []
    teams = ["India", "Australia", "England", "Pakistan", "South Africa",
            "New Zealand", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan",
            "Mumbai Indians", "Chennai Super Kings", "Royal Challengers Bengaluru",
            "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
            "Punjab Kings", "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants"]

    for match_id in range(500):
        team_a, team_b = np.random.choice(teams, 2, replace=False)
        toss_winner = np.random.choice([team_a, team_b])
        toss_decision = np.random.choice(["bat", "field"])
        venue = np.random.choice(["home", "away", "neutral"])
        format_ = np.random.choice(["T20", "ODI", "IPL", "Test"])

        if format_ in ["T20", "IPL"]:
            total_overs = 20
        elif format_ == "ODI":
            total_overs = 50
        else:
            total_overs = 90

        team_a_score = int(np.random.normal(160, 25)) if format_ == "T20" else int(np.random.normal(270, 40))
        team_b_score = int(np.random.normal(155, 25)) if format_ == "T20" else int(np.random.normal(265, 40))
        team_a_score = max(80, team_a_score)
        team_b_score = max(80, team_b_score)

        team_a_wickets = np.random.randint(3, 11)
        team_b_wickets = np.random.randint(3, 11)

        overs_completed = round(np.random.uniform(total_overs * 0.5, total_overs), 1)
        overs_remaining = round(total_overs - overs_completed, 1)
        balls_remaining = int(overs_remaining * 6)
        runs_needed = max(0, team_a_score - team_b_score + np.random.randint(1, 20))
        required_run_rate = round(runs_needed / max(overs_remaining, 0.1), 2)
        current_run_rate = round(team_b_score / max(overs_completed, 0.1), 2)
        run_rate_diff = round(current_run_rate - required_run_rate, 2)
        wickets_remaining = 10 - team_b_wickets
        balls_per_wicket = round(overs_completed * 6 / max(team_b_wickets, 1), 2)
        result = 1 if team_a_score > team_b_score else 0

        records.append({
            "match_id": match_id,
            "team_a": team_a,
            "team_b": team_b,
            "format": format_,
            "venue": venue,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
            "team_a_score": team_a_score,
            "team_b_score": team_b_score,
            "team_a_wickets": team_a_wickets,
            "team_b_wickets": team_b_wickets,
            "team_a_run_rate": round(team_a_score / total_overs, 2),
            "team_b_run_rate": round(team_b_score / max(overs_completed, 0.1), 2),
            "overs_completed": overs_completed,
            "overs_remaining": overs_remaining,
            "balls_remaining": balls_remaining,
            "required_run_rate": required_run_rate,
            "current_run_rate": current_run_rate,
            "run_rate_diff": run_rate_diff,
            "wickets_remaining": wickets_remaining,
            "balls_per_wicket": balls_per_wicket,
            "result": result
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    print("Generating batting data...")
    generate_batting_data().to_csv("data/raw/batting_stats.csv", index=False)
    print("Generating bowling data...")
    generate_bowling_data().to_csv("data/raw/bowling_stats.csv", index=False)
    print("Generating match data...")
    generate_match_data().to_csv("data/raw/match_results.csv", index=False)
    print("All datasets generated!")