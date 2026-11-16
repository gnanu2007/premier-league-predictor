import pandas as pd
from datetime import datetime
import requests
import streamlit as st
import joblib

from src.data_loader import load_data
from src.features import build_match_features

st.set_page_config(
    page_title="Premier League Predictor",
    page_icon="⚽",
    layout="wide"
)

# Custom dashboard design
st.markdown("""
<style>
.stApp {
    background-color: #101827;
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
.hero {
    background: linear-gradient(120deg, #162b50, #174b3b);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    margin-bottom: 25px;
}
.hero h1 {
    font-size: 38px;
    margin-bottom: 8px;
}
.hero p {
    color: #d1fae5 !important;
}
.metric-card {
    background: #1e293b;
    padding: 22px;
    border-radius: 14px;
    text-align: center;
    border: 1px solid #334155;
}
.metric-card h2 {
    font-size: 30px;
    color: #4ade80 !important;
}
div.stButton > button {
    background-color: #16a34a;
    color: white;
    border-radius: 10px;
    border: none;
    height: 48px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}
div.stButton > button:hover {
    background-color: #15803d;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def get_data():
    return load_data()

df = get_data()
@st.cache_data(ttl=86400)
def get_team_logo(team_name):
    # Handle common team-name differences
    aliases = {
        "Man City": "Manchester City",
        "Man United": "Manchester United",
        "Spurs": "Tottenham Hotspur",
        "Wolves": "Wolverhampton Wanderers",
        "Leicester": "Leicester City",
        "Newcastle": "Newcastle United",
        "West Ham": "West Ham United",
        "Brighton": "Brighton",
    }

    search_name = aliases.get(team_name, team_name)

    try:
        url = "https://www.thesportsdb.com/api/v1/json/123/searchteams.php"
        response = requests.get(
            url,
            params={"t": search_name},
            timeout=10
        )
        response.raise_for_status()

        teams_data = response.json().get("teams")

        if teams_data:
            return teams_data[0].get("strBadge")

    except requests.RequestException:
        return None

    return None
model = joblib.load("models/model.pkl")

teams = sorted(
    set(df["HomeTeam"]) | set(df["AwayTeam"])
)

# Dashboard heading
st.markdown("""
<div class="hero">
    <h1>⚽ PREMIER LEAGUE PREDICTOR</h1>
    <p>Machine Learning Powered Football Match Analysis</p>
</div>
""", unsafe_allow_html=True)

# Match selection
st.subheader("🏆 Select Your Match")

col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("🏠 Home Team", teams)

with col2:
    away_options = [team for team in teams if team != home_team]
    away_team = st.selectbox("✈️ Away Team", away_options)
    # Display selected team logos
logo_col1, logo_col2, logo_col3 = st.columns([2, 1, 2])

with logo_col1:
    home_logo = get_team_logo(home_team)

    if home_logo:
        st.image(home_logo, width=120)
    else:
        st.write("⚽ Logo unavailable")

    st.markdown(f"**{home_team}**")

with logo_col2:
    st.markdown("## VS")

with logo_col3:
    away_logo = get_team_logo(away_team)

    if away_logo:
        st.image(away_logo, width=120)
    else:
        st.write("⚽ Logo unavailable")

    st.markdown(f"**{away_team}**")

st.write("")
# Store prediction history during the current session
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if st.button("⚽ PREDICT MATCH"):

    X = build_match_features(df, home_team, away_team)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    result_names = {
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win"
    }

    st.markdown("---")
    st.subheader("📊 Match Prediction")

    result_col, home_col, draw_col, away_col = st.columns(4)

    with result_col:
        st.markdown(
            f'<div class="metric-card"><h3>Predicted Result</h3>'
            f'<h2>{result_names[prediction]}</h2></div>',
            unsafe_allow_html=True
        )

    probability_map = dict(zip(model.classes_, probabilities))

    # Save prediction
    st.session_state.prediction_history.append({
        "Date": datetime.now().strftime("%d-%m-%Y %H:%M"),
        "Home Team": home_team,
        "Away Team": away_team,
        "Predicted Result": result_names[prediction],
        "Home Win %": round(float(probability_map["H"]) * 100, 2),
        "Draw %": round(float(probability_map["D"]) * 100, 2),
        "Away Win %": round(float(probability_map["A"]) * 100, 2)
    })

    with home_col:
        st.metric("🏠 Home Win", f"{probability_map['H'] * 100:.2f}%")

    with draw_col:
        st.metric("🤝 Draw", f"{probability_map['D'] * 100:.2f}%")

    with away_col:
        st.metric("✈️ Away Win", f"{probability_map['A'] * 100:.2f}%")

    st.markdown("---")
    st.subheader("📈 Probability Analysis")

    for result in ["H", "D", "A"]:
        name = result_names[result]
        probability = float(probability_map[result])

        st.write(f"**{name} — {probability * 100:.2f}%**")
        st.progress(probability)

# Prediction History
st.markdown("---")
st.header("📜 Prediction History")

if st.session_state.prediction_history:

    history_df = pd.DataFrame(st.session_state.prediction_history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    # Prediction History Chart
    st.subheader("📊 Prediction Results Chart")

    chart_data = history_df["Predicted Result"].value_counts()

    st.bar_chart(chart_data)

    # Best Prediction Summary
    st.markdown("---")
    st.subheader("🏆 Prediction Summary")

    total_predictions = len(history_df)

    home_wins = (history_df["Predicted Result"] == "Home Win").sum()
    draws = (history_df["Predicted Result"] == "Draw").sum()
    away_wins = (history_df["Predicted Result"] == "Away Win").sum()

    most_predicted = history_df["Predicted Result"].value_counts().idxmax()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Predictions", total_predictions)
    col2.metric("🏠 Home Wins", home_wins)
    col3.metric("🤝 Draws", draws)
    col4.metric("✈️ Away Wins", away_wins)

    st.success(f"Most frequently predicted result: {most_predicted}")

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.rerun()

else:
    st.info("No predictions yet. Predict a match to see your history!")

# Footer
st.markdown("---")
st.caption(
    "Premier League Predictor | Built using Python, "
    "Pandas, Scikit-learn and Streamlit"
)
st.caption(
    "Predictions are estimates based on historical data "
    "and are not guaranteed results."
)
