# 🏏 CricketIQ — AI-Powered Cricket Intelligence

> Predict. Analyze. Dominate.

CricketIQ is a deep learning-powered cricket analytics platform that predicts batting performance, bowling performance, and live match outcomes using LSTM neural networks.

---

## 🎯 Features

- **Batting Predictor** — Predicts runs and strike rate based on last 10 matches
- **Bowling Predictor** — Predicts wickets and economy rate based on last 10 matches
- **Live Match Outcome** — Predicts win probability with real-time RRR, CRR, balls remaining
- **Multi-Format Support** — T20, ODI, IPL, Test cricket
- **IPL + International Teams** — All 10 IPL teams + 10 international teams

---

## 🧠 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| PyTorch | LSTM neural networks |
| Scikit-learn | Preprocessing & scaling |
| Streamlit | Web application UI |
| Plotly | Interactive charts |
| Pandas & NumPy | Data handling |

---

## 🏗️ Architecture

CricketIQ uses a 3-model LSTM architecture:
- **Batting Model** — Trained on 10-match sequences for runs & strike rate
- **Bowling Model** — Trained on 10-match sequences for wickets & economy
- **Match Outcome Model** — Real-time prediction using live match state

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install streamlit torch scikit-learn plotly pandas numpy
