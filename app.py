import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Import functions and data from the main script
from main import (
    create_game_data,
    engineer_features,
    create_target_variable,
    team_season_stats
)

# --- Model Training and Caching ---

@st.cache_resource
def train_model():
    """
    Trains the logistic regression model and returns all necessary components.
    The results are cached to prevent re-training on every app interaction.
    """
    # Create the dataset for training
    games_df = create_game_data(team_season_stats, n_games_per_season=100)
    games_df = engineer_features(games_df)
    games_df = create_target_variable(games_df)

    # Define features and target
    feature_columns_for_model = [
        'home_win_pct', 'away_win_pct', 'win_pct_diff', 'ppg_diff',
        'fg_pct_diff', 'home_team_strong', 'away_team_strong'
    ]
    X = games_df[feature_columns_for_model]
    y = games_df['home_team_win']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win'], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns_for_model,
        'coefficient': model.coef_[0]
    }).sort_values(by='coefficient', ascending=False)

    return model, scaler, accuracy, report, cm, feature_importance, feature_columns_for_model, team_season_stats

# --- UI Pages ---

def show_home_page():
    """Displays the welcome page of the app."""
    st.image("basketball.jpg", use_column_width=True)
    st.title("HomeCourt AI: NBA Game Predictor")
    
    st.markdown("""
        **Welcome to HomeCourt AI, your MVP for NBA game predictions!**
        
        This application leverages a robust dataset of historical team and player performance to forecast game outcomes with a data-driven edge. Move beyond simple win-loss records and dive into deeper analytics.
        
        ### What you can do:
        - **🔮 Make Predictions:** Select two teams and get an instant prediction for their matchup, complete with win probabilities.
        - **📊 Explore Model Performance:** See how our prediction model performs and understand the key factors driving its decisions.
        
        *This is an MVP built for demonstration. The model uses simulated game data based on real season stats to train.*
    """)

    st.divider()

    st.header("Powered by Comprehensive NBA Data")
    st.markdown("""
        Our predictions aren't just based on wins and losses. We leverage a rich dataset spanning decades of NBA history, including both traditional and advanced statistics. This depth of data allows our model to capture nuanced team strengths and weaknesses.
        
        Here's a glimpse of the types of stats we use:
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Core Player Stats**")
        st.markdown("""
            - Points Per Game (PPG)
            - Field Goal % (FG%)
            - 3-Point % (3P%)
            - Rebounds Per Game (RPG)
            - Assists Per Game (APG)
        """)

    with col2:
        st.info("**Advanced Analytics**")
        st.markdown("""
            - Player Efficiency Rating (PER)
            - True Shooting % (TS%)
            - Win Shares (WS)
            - Box Plus/Minus (BPM)
            - Usage Percentage (USG%)
        """)

    with col3:
        st.info("**Team & Opponent Data**")
        st.markdown("""
            - Team Win/Loss Records
            - Points Per 100 Possessions
            - Opponent Field Goal %
            - Turnover Percentage (TOV%)
            - Pace Factor
        """)
    
    st.markdown("""
        *...and many more, giving our model a holistic view of team performance.*
    """)

def show_performance_page(accuracy, report, cm, feature_importance):
    """Displays the model's performance metrics."""
    st.header("Model Performance Evaluation")
    st.markdown("Here's how our logistic regression model performs on the test dataset.")

    st.metric("Model Accuracy", f"{accuracy:.2%}")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    with col2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Predicted Away Win', 'Predicted Home Win'],
                    yticklabels=['Actual Away Win', 'Actual Home Win'])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)

    st.subheader("Feature Importance")
    st.markdown("These are the factors that most influence the model's predictions. Positive coefficients increase the likelihood of a home team win, while negative coefficients decrease it.")
    st.bar_chart(feature_importance.set_index('feature'))


def show_prediction_page(model, scaler, feature_columns, all_teams_stats):
    """Allows users to select teams and get a game prediction."""
    st.header("🔮 Make a Prediction")
    st.markdown("Select a home and away team to see who the model thinks will win.")

    # Filter teams to only show those with 2025 season data
    teams_2025 = all_teams_stats[all_teams_stats['season'] == 2025]['team'].unique()
    teams = sorted(teams_2025)
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Home Team", teams, index=0)
    with col2:
        away_team = st.selectbox("Away Team", teams, index=1)

    if home_team == away_team:
        st.error("Please select two different teams.")
        return

    if st.button("Predict Outcome", type="primary"):
        # Get the latest season stats for the selected teams
        latest_season = all_teams_stats['season'].max()
        home_stats = all_teams_stats[(all_teams_stats['team'] == home_team) & (all_teams_stats['season'] == latest_season)].iloc[0]
        away_stats = all_teams_stats[(all_teams_stats['team'] == away_team) & (all_teams_stats['season'] == latest_season)].iloc[0]

        # Create a single game record
        game_record = {
            'home_team': home_team, 'away_team': away_team,
            'home_wins': home_stats.get('w', 0), 'home_losses': home_stats.get('l', 0),
            'away_wins': away_stats.get('w', 0), 'away_losses': away_stats.get('l', 0),
            'home_ppg': home_stats.get('pts', 100) / home_stats.get('g', 82),
            'away_ppg': away_stats.get('pts', 100) / away_stats.get('g', 82),
            'home_fg_pct': home_stats.get('fg_percent', 0.45),
            'away_fg_pct': away_stats.get('fg_percent', 0.45),
        }
        
        # Engineer features for the prediction
        game_df = pd.DataFrame([game_record])
        features_df = engineer_features(game_df)
        
        # Select and scale features
        X_pred = features_df[feature_columns]
        X_pred_scaled = scaler.transform(X_pred)
        
        # Make prediction
        prediction_proba = model.predict_proba(X_pred_scaled)[0]
        home_win_prob = prediction_proba[1]

        st.subheader("Prediction Result")
        if home_win_prob > 0.5:
            st.success(f"**{home_team}** is predicted to win.")
        else:
            st.success(f"**{away_team}** is predicted to win.")
        
        st.metric(f"{home_team} Win Probability", f"{home_win_prob:.2%}")
        
        st.expander("See Prediction Details").write({
            "Home Team Win Probability": f"{home_win_prob:.2%}",
            "Away Team Win Probability": f"{prediction_proba[0]:.2%}",
            "Input Features": features_df[feature_columns].iloc[0].to_dict()
        })


# --- Main App ---

def main():
    st.set_page_config(page_title="HomeCourt AI", page_icon="🏀", layout="wide")
    
    # Load model and data
    model, scaler, accuracy, report, cm, feature_importance, feature_columns, all_teams_stats = train_model()

    st.sidebar.title("🏀 Navigation")
    page = st.sidebar.radio(
        "Choose a section:",
        ["Home", "Model Performance", "Make Predictions"]
    )

    if page == "Home":
        show_home_page()
    elif page == "Model Performance":
        show_performance_page(accuracy, report, cm, feature_importance)
    elif page == "Make Predictions":
        show_prediction_page(model, scaler, feature_columns, all_teams_stats)

if __name__ == "__main__":
    main()
