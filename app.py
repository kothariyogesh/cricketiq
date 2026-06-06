import streamlit as st
import numpy as np
import pickle
import torch
import torch.nn as nn
import plotly.graph_objects as go

st.set_page_config(page_title="CricketIQ", page_icon="🏏", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Inter:wght@300;400;500&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff, #00ff87);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #00d4ff33;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}
.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-label { color: #aaa; font-size: 0.85rem; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)


class CricketLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super(CricketLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class MatchClassifier(nn.Module):
    def __init__(self, input_size):
        super(MatchClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = self.relu(self.fc3(out))
        out = self.sigmoid(self.fc4(out))
        return out


@st.cache_resource
def load_models():
    try:
        bat_runs = CricketLSTM(input_size=9)
        bat_runs.load_state_dict(torch.load("models/batting_runs.pth", map_location="cpu"))
        bat_runs.eval()

        bat_sr = CricketLSTM(input_size=9)
        bat_sr.load_state_dict(torch.load("models/batting_sr.pth", map_location="cpu"))
        bat_sr.eval()

        bowl_wk = CricketLSTM(input_size=9)
        bowl_wk.load_state_dict(torch.load("models/bowling_wickets.pth", map_location="cpu"))
        bowl_wk.eval()

        bowl_eco = CricketLSTM(input_size=9)
        bowl_eco.load_state_dict(torch.load("models/bowling_economy.pth", map_location="cpu"))
        bowl_eco.eval()

        match = MatchClassifier(input_size=20)
        match.load_state_dict(torch.load("models/match_outcome.pth", map_location="cpu"))
        match.eval()

        bat_scaler = pickle.load(open("models/batting_scaler.pkl", "rb"))
        bowl_scaler = pickle.load(open("models/bowling_scaler.pkl", "rb"))
        match_scaler = pickle.load(open("models/match_scaler.pkl", "rb"))

        return {
            "bat_runs": bat_runs,
            "bat_sr": bat_sr,
            "bowl_wk": bowl_wk,
            "bowl_eco": bowl_eco,
            "match": match,
            "bat_scaler": bat_scaler,
            "bowl_scaler": bowl_scaler,
            "match_scaler": match_scaler
        }, True
    except Exception as e:
        return {}, False


models, loaded = load_models()

st.markdown('<h1 class="main-title">🏏 CricketIQ</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;color:#888;">AI-Powered Cricket Intelligence · LSTM · PyTorch · Built by Yogesh Kothari</p>', unsafe_allow_html=True)

if not loaded:
    st.error("Models not found. Please run: python train_all.py")
    st.stop()

st.success("✅ All models loaded!")

with st.sidebar:
    st.markdown("### 🎯 Select Mode")
    mode = st.radio("", ["🏏 Batting", "🎳 Bowling", "🏆 Match Outcome"])
    st.markdown("---")
    st.markdown("**Built by Yogesh Kothari**")
    st.markdown("MCA · AI & Deep Learning · Zonal Cricketer")


if mode == "🏏 Batting":
    st.subheader("Batting Performance Predictor")
    batting_data = []
    for i in range(10):
        with st.expander(f"Match {i+1}", expanded=(i == 0)):
            c1, c2, c3, c4 = st.columns(4)
            runs = c1.number_input("Runs", 0, 250, 0, key=f"r{i}")
            balls = c2.number_input("Balls", 0, 200, 0, key=f"b{i}")
            fours = c3.number_input("4s", 0, 30, 0, key=f"f{i}")
            sixes = c4.number_input("6s", 0, 20, 0, key=f"s{i}")
            c5, c6 = st.columns(2)
            opp = c5.selectbox("Opposition", ["weak", "medium", "strong"], key=f"o{i}")
            venue = c6.selectbox("Venue", ["home", "away", "neutral"], key=f"v{i}")
            sr = round((runs / max(balls, 1)) * 100, 2)
            opp_enc = {"weak": 2, "medium": 1, "strong": 0}[opp]
            venue_enc = {"home": 0, "away": 1, "neutral": 2}[venue]
            batting_data.append([runs, balls, sr, fours, sixes, opp_enc, venue_enc, 0, 0.0])

    if st.button("🔮 Predict Next Innings"):
        seq = np.array(batting_data)
        seq_scaled = models["bat_scaler"].transform(seq)
        seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0)
        with torch.no_grad():
            pred_runs = max(0, models["bat_runs"](seq_tensor).item())
            pred_sr = max(0, models["bat_sr"](seq_tensor).item())
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{pred_runs:.0f}</div><div class="metric-label">Predicted Runs</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{pred_sr:.1f}</div><div class="metric-label">Predicted Strike Rate</div></div>', unsafe_allow_html=True)
        runs_history = [d[0] for d in batting_data]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, 11)), y=runs_history, mode="lines+markers", name="History", line=dict(color="#00d4ff")))
        fig.add_trace(go.Scatter(x=[11], y=[pred_runs], mode="markers", name="Predicted", marker=dict(color="#00ff87", size=14, symbol="star")))
        fig.update_layout(title="Run Trend + Prediction", template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)


elif mode == "🎳 Bowling":
    st.subheader("Bowling Performance Predictor")
    bowling_data = []
    for i in range(10):
        with st.expander(f"Match {i+1}", expanded=(i == 0)):
            c1, c2, c3, c4 = st.columns(4)
            overs = c1.number_input("Overs", 0.0, 10.0, 0.0, key=f"ov{i}")
            wickets = c2.number_input("Wickets", 0, 7, 0, key=f"wk{i}")
            runs_c = c3.number_input("Runs Given", 0, 100, 0, key=f"rc{i}")
            dots = c4.number_input("Dot Balls", 0, 30, 0, key=f"db{i}")
            c5, c6 = st.columns(2)
            opp = c5.selectbox("Opposition", ["weak", "medium", "strong"], key=f"bo{i}")
            venue = c6.selectbox("Venue", ["home", "away", "neutral"], key=f"bv{i}")
            eco = round(runs_c / max(overs, 0.1), 2)
            opp_enc = {"weak": 2, "medium": 1, "strong": 0}[opp]
            venue_enc = {"home": 0, "away": 1, "neutral": 2}[venue]
            bowling_data.append([overs, wickets, runs_c, eco, dots, opp_enc, venue_enc, 0, 0.0])

    if st.button("🔮 Predict Next Spell"):
        seq = np.array(bowling_data)
        seq_scaled = models["bowl_scaler"].transform(seq)
        seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0)
        with torch.no_grad():
            pred_wk = max(0, models["bowl_wk"](seq_tensor).item())
            pred_eco = max(0, models["bowl_eco"](seq_tensor).item())
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{pred_wk:.1f}</div><div class="metric-label">Predicted Wickets</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{pred_eco:.2f}</div><div class="metric-label">Predicted Economy</div></div>', unsafe_allow_html=True)


else:
    st.subheader("Match Outcome Predictor")
    st.markdown("Enter live match situation for prediction")

    international_teams = ["India", "Australia", "England", "Pakistan", "South Africa",
                           "New Zealand", "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan"]

    ipl_teams = ["Royal Challengers Bengaluru", "Chennai Super Kings", "Mumbai Indians", "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
                 "Kolkata Knight Riders", "Delhi Capitals", "Rajasthan Royals",
                 "Punjab Kings", "Sunrisers Hyderabad", "Gujarat Titans", "Lucknow Super Giants"]

    # Format must be selected FIRST
    fmt = st.selectbox("Format", ["T20", "ODI", "IPL", "Test"])

    # Then teams based on format
    if fmt == "IPL":
        teams = ipl_teams
    else:
        teams = international_teams

    team_enc = {t: i for i, t in enumerate(sorted(international_teams + ipl_teams))}

    c1, c2, c3 = st.columns(3)
    with c1:
        team_a = st.selectbox("Team A (Batting First)", teams)
        team_a_score = st.number_input("Team A Score", 0, 500, 0)
        team_a_wk = st.number_input("Team A Wickets", 0, 10, 0)
    with c2:
        team_b = st.selectbox("Team B (Chasing)", [t for t in teams if t != team_a])
        team_b_score = st.number_input("Team B Score", 0, 500, 0)
        team_b_wk = st.number_input("Team B Wickets", 0, 10, 0)
    with c3:
        venue = st.selectbox("Venue", ["home", "away", "neutral"])
        toss_winner = st.selectbox("Toss Winner", [team_a, team_b])
        toss_decision = st.selectbox("Toss Decision", ["bat", "field"])

    st.markdown("### 📊 Live Match Context")
    c4, c5, c6 = st.columns(3)
    with c4:
        if fmt in ["T20", "IPL"]:
            total_overs = 20
        elif fmt == "ODI":
            total_overs = 50
        else:
            total_overs = 90
        overs_completed = c4.number_input("Overs Completed", 0.0, float(total_overs), 0.0)
    with c5:
        overs_remaining = round(total_overs - overs_completed, 1)
        st.metric("Overs Remaining", overs_remaining)
    with c6:
        balls_remaining = int(overs_remaining * 6)
        st.metric("Balls Remaining", balls_remaining)

    runs_needed = max(0, team_a_score - team_b_score)
    required_run_rate = round(runs_needed / max(overs_remaining, 0.1), 2)
    current_run_rate = round(team_b_score / max(overs_completed, 0.1), 2)
    run_rate_diff = round(current_run_rate - required_run_rate, 2)
    wickets_remaining = 10 - team_b_wk
    balls_per_wicket = round((overs_completed * 6) / max(team_b_wk, 1), 2)

    c7, c8, c9 = st.columns(3)
    c7.metric("Required RR", required_run_rate)
    c8.metric("Current RR", current_run_rate)
    c9.metric("Run Rate Diff", run_rate_diff)

    if st.button("🔮 Predict Result"):
        features = np.array([[
            team_enc.get(team_a, 0), team_enc.get(team_b, 1),
            team_enc.get(toss_winner, 0),
            {"home": 0, "away": 1, "neutral": 2}[venue],
            {"T20": 1, "ODI": 0}[fmt],
            {"bat": 0, "field": 1}[toss_decision],
            team_a_score, team_b_score,
            team_a_wk, team_b_wk,
            round(team_a_score / total_overs, 2),
            round(team_b_score / max(overs_completed, 0.1), 2),
            overs_completed, overs_remaining, balls_remaining,
            required_run_rate, current_run_rate, run_rate_diff,
            wickets_remaining, balls_per_wicket
        ]])

        features_scaled = models["match_scaler"].transform(features)
        tensor = torch.FloatTensor(features_scaled)
        with torch.no_grad():
            prob = models["match"](tensor).item()

        team_a_prob = round(prob * 100, 1)
        team_b_prob = round((1 - prob) * 100, 1)
        winner = team_a if prob > 0.5 else team_b

        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{team_a_prob}%</div><div class="metric-label">{team_a}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">🏆</div><div class="metric-label">Winner: {winner}</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value">{team_b_prob}%</div><div class="metric-label">{team_b}</div></div>', unsafe_allow_html=True)

        fig = go.Figure(go.Pie(
            labels=[team_a, team_b],
            values=[team_a_prob, team_b_prob],
            hole=0.6,
            marker_colors=["#00d4ff", "#00ff87"]
        ))
        fig.update_layout(title="Win Probability", template="plotly_dark", height=350)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown('<p style="text-align:center;color:#555;">CricketIQ · Yogesh Kothari · MCA · AI Engineer · Zonal Cricketer · Pune</p>', unsafe_allow_html=True)


