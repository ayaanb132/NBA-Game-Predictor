import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Load NBA data files and combine team stats from multiple sources
player_stats = pd.read_csv('nba_data/Player Per Game.csv')

team_summaries = pd.read_csv('nba_data/Team Summaries.csv')
team_totals = pd.read_csv('nba_data/Team Totals.csv')
team_totals_subset = team_totals[['season', 'team', 'lg', 'g', 'pts', 'fg_percent']]

# Combine team summary data with team totals data 
team_stats = pd.merge(
    team_summaries,
    team_totals_subset,
    on=['season','team','lg'],
    how='inner'
)

player_totals = pd.read_csv('/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data/Player Totals.csv')

# Show basic info about our datasets

print("Player Stats Shape:", player_stats.shape)
print("Team Stats Shape:", team_stats.shape)
print("Player Totals Shape:", player_totals.shape)

# Look at the first few rows of team data

print("\nTeam Stats - First 5 rows:")
print(team_stats.head())

# Check data types and count missing values
print("\nTeam Stats - Column Information:")
print(team_stats.info())

# Get basic statistics for numerical columns
print("\nTeam Stat - Basic Statistics")
print(team_stats.describe())






# Check team data structure and get ready to work with it

# Check what columns we have available
print("Team Stats Columns:")
print(team_stats.columns.tolist())

# Look at what seasons and teams we have in our data

print("\nUnique values in key columns:")
if 'season' in team_stats.columns:
    print("Seasons available:", team_stats['season'].nunique())
    print("Season range:",team_stats['season'].min(), "to", team_stats['season'].max())
if 'team' in team_stats.columns:
    print("Teams:", team_stats['team'].unique()[:10]) # Show first 10 teams as examples

team_season_stats =team_stats.copy()
print("\nWorking with team season statistics")
print("Sample of team season data:")
print(team_season_stats.head())

# Clean up the data by fixing missing values and removing unnecessary columns

# Find missing data points

print("Missing values per column:")
missing_values = team_season_stats.isnull().sum()
print(missing_values[missing_values > 0])

# Fill in missing values with the median (middle value) for each column

numerical_columns = team_season_stats.select_dtypes(include=[np.number]).columns

for col in numerical_columns:
    if team_season_stats[col].isnull().sum() > 0:
        median_value = team_season_stats[col].median()
        team_season_stats[col].fillna(median_value, inplace=True)
        print(f"Filled {col} missing values with median: {median_value}")

# Make sure data types look correct
print("\nData types:")
print(team_season_stats.dtypes)

# Clean up team names by removing extra spaces

if 'team' in team_season_stats.columns:
    team_season_stats['team'] = team_season_stats['team'].str.strip()


columns_to_remove = []

# Remove columns not needed for prediction
manual_remove = ['lg','abbreviation','playoffs','arena','attend','attend_g']

# remove any unnamed index columns
for col in team_season_stats.columns:
    if col.lower() in ['unnamed','index','rk'] or 'unnamed' in col.lower():
        columns_to_remove.append(col)
columns_to_remove.extend(manual_remove)

columns_to_remove = [col for col in columns_to_remove if col in team_season_stats.columns]
team_season_stats.drop(columns=columns_to_remove, inplace=True)

print(f"\nCleaned dataset shape: {team_season_stats.shape}")





def create_game_data(season_stats, n_games_per_season=50):
    """
    -------------------------------------------------------
    Takes team season stats and creates fake game matchups.
    Randomly pairs teams together and pulls their stats to simulate games.
    Use: games_df = create_game_data(season_stats, n_games_per_season)
    -------------------------------------------------------
    Parameters:
        season_stats - dataframe containing team season statistics (DataFrame)
        n_games_per_season - number of games to simulate per season (int, default=50)
    Returns:
        games_df - dataframe with home/away team info and their performance stats (DataFrame)
    -------------------------------------------------------
    """
    games = []

    # Get all unique teams and seasons from data

    teams = season_stats['team'].unique()
    seasons = season_stats['season'].unique() if 'season' in season_stats.columns else [2025]

    for season in seasons:
        season_data = season_stats[season_stats['season'] == season] if 'season' in season_stats.columns else season_stats

        # Make random game matchups for this season
        for i in range(n_games_per_season):
            if len(season_data) < 2:
                continue
    
            # Pick two different teams to play each other
            home_away_teams = np.random.choice(season_data['team'].values,2,replace=False)
            home_team = home_away_teams[0]
            away_team = home_away_teams[1]

            # Look up the season stats for both teams
            home_stats = season_data[season_data['team'] == home_team].iloc[0]
            away_stats = season_data[season_data['team']== away_team].iloc[0]

            game_record = {
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'home_wins': home_stats.get('w',0), # Get wins, use 0 if not found
                'home_losses': home_stats.get('l',0),
                'away_wins': away_stats.get('w',0),
                'away_losses': away_stats.get('l',0),
                'home_ppg': home_stats.get('pts',100)/home_stats.get('g',82),
                'away_ppg': away_stats.get('pts',100)/away_stats.get('g',82),
                'home_fg_pct': home_stats.get('fg_percent',0.45),
                'away_fg_pct': away_stats.get('fg_percent',0.45),
            }

            games.append(game_record)

    return pd.DataFrame(games)


def engineer_features(games_df):
    """
    -------------------------------------------------------
    Takes the game data and calculates useful stats for prediction.
    Adds win percentages, point differences, and team strength indicators.
    Use: enhanced_df = engineer_features(games_df)
    -------------------------------------------------------
    Parameters:
        games_df - dataframe containing game matchup data (DataFrame)
    Returns:
        enhanced_df - dataframe with additional features that help the model 
            understand which team is stronger (DataFrame)
    -------------------------------------------------------
    """

    df = games_df.copy()


    # Calculate each team's win percentage

    df['home_win_pct'] = df['home_wins'] / (df['home_wins'] + df['home_losses'])
    df['away_win_pct'] = df['away_wins'] / (df['away_wins'] + df['away_losses'])

    # If team has no games played, assume 50% win rate
    df['home_win_pct'] = df['home_win_pct'].fillna(0.5) 
    df['away_win_pct'] = df['away_win_pct'].fillna(0.5)

    # Create difference features (how much better is home team than away team)

    df['win_pct_diff'] = df['home_win_pct'] - df['away_win_pct']
    df['ppg_diff'] = df['home_ppg'] - df['away_ppg']
    df['fg_pct_diff'] = df['home_fg_pct'] - df['away_fg_pct']

    # Add home court advantage flag (always 1 for home team)
    df['home_court_advantage'] = 1

    # Mark teams as "strong" if they win more than 55% of games
    df['home_team_strong'] = (df['home_win_pct']>0.55).astype(int)
    df['away_team_strong'] = (df['away_win_pct']>0.55).astype(int)

    return df




def create_target_variable(df):
    """
    -------------------------------------------------------
    Creates realistic win/loss outcomes for each game.
    Uses team stats to calculate win probability, then randomly generates
    actual game results based on those probabilities.
    Use: df_with_outcomes = create_target_variable(df)
    -------------------------------------------------------
    Parameters:
        df - dataframe containing game data with features (DataFrame)
    Returns:
        df_with_outcomes - dataframe with added 'home_team_win' column 
            containing binary outcomes (DataFrame)
    -------------------------------------------------------
    """

    # Calculate home team win probability using weighted team stats

    home_win_prob = (
        0.50 + # Start with 50% base probability for home team
        0.30 * df['win_pct_diff'] + # Win percentage difference has big impact
        0.15 * (df['ppg_diff'] /20) + # Points per game difference (scaled down)
        0.05 * df['fg_pct_diff'] # Shooting percentage difference has small impact
    )

    # Keep probabilities realistic (between 10% and 90%)

    home_win_prob = np.clip(home_win_prob,0.1,0.9)

    # Generate random win/loss outcomes based on calculated probabilities

    np.random.seed(42) # Set random seed so results are the same each time
    df['home_team_win'] = np.random.binomial(1, home_win_prob)

    return df

# Create our target variable (what we want to predict)

def main():
    """ Main execution function - runs all the analysis"""

    #build simulated game-set

    print("creating game-level data from season statistics...")
    games_df = create_game_data(team_season_stats,n_games_per_season=100)
    print(f"Created {len(games_df)} game records")
    print("\nSample game data")
    print(games_df.head())

    print("Engineering features...")
    games_df=engineer_features(games_df)

    # show new features created

    print("Engineered features")
    feature_columns = ['home_win_pct', 'away_win_pct','win_pct_diff','fg_pct_diff','home_team_strong','away_team_strong']
    print(games_df[feature_columns].head())

    #create target variable to predict

    print("Creating target varibale")
    games_df = create_target_variable(games_df)

    # see how often home teams win in simulated data

    print("Home team win distribution:")
    print(games_df['home_team_win'].value_counts())
    print(f"Home team win rate: {games_df['home_team_win'].mean() * 100:.3f}")

    correlation_with_target = games_df[feature_columns + ['home_team_win']].corr()['home_team_win'].sort_values(ascending=False)
    print("\nCorrelation with home team wins:")
    print(correlation_with_target)

    # build and train ML model

    #set up data for ML model
    feature_columns_for_model = ['home_win_pct','away_win_pct,win_pct_diff','ppg_diff','fg_pct_diff','home_team_strong','away_team_strong']

    X = games_df[feature_columns_for_model]
    y = games_df['home_team_win']

    print("features for modeling:")
    print(X.head())
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    #split data into 80/20 training and testing

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    print(f"\nTraining set size: {X_train.shape[0]} games")
    print(f"Testing set size: {X_test.shape[0]} games")
    print(f"Training set home win rate: {y_train.mean():.3f}")
    print(f"Testing set home win rate: {y_test.mean():.3f}")

    # scale features 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed")
    print("Training features mean after scaling: ", X_train_scaled.mean(axis=0.round(3))))


          

















print("Creating target varibale...")
games_df = create_target_variable(games_df)

# See how often home teams win in our fake data

print("Home team win distribution:")
print(games_df['home_team_win'].value_counts())
print(f"Home team win rate: {games_df['home_team_win'].mean() * 100:.3f}%")

correlation_with_target = games_df[feature_columns + ['home_team_win']].corr()['home_team_win'].sort_values(ascending=False)
print("\nCorrelation with home team wins:")
print(correlation_with_target)


# Build and train the machine learning model

# Set up the data for our machine learning model

feature_columns_for_model = ['home_win_pct', 'away_win_pct', 'win_pct_diff','ppg_diff','fg_pct_diff']

X = games_df[feature_columns_for_model]
y = games_df['home_team_win']

print("Features for modeling:")
print(X.head())
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# Split data into training (80%) and testing (20%) sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} games")
print(f"Testing set size: {X_test.shape[0]} games")
print(f"Training set home win rate: {y_train.mean():.3f}")
print(f"Testing set home win rate: {y_test.mean():.3f}")

# Scale features so they're all on the same scale (helps the model train better)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed")
print("Training features mean after scaling: ", X_train_scaled.mean(axis=0).round(3))
print("Training features std after scaling: ", X_train_scaled.std(axis=0).round(3))

# Create and train a logistic regression model

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled,y_train)

print("\nModel training completed!")

# Test the trained model on data it hasn't seen before

y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:,1] # Get probability scores for home team wins

print("Predicitons generated")
print(f"Predicted home wins: {y_pred.sum()}")
print(f"Actual home wins: {y_test.sum()}")

# Test the model and see how well it predicts game outcomes

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")

print("                 Predicted Away Win   Predicted Home Win") 
print(f"Actual Away Win  {cm[0, 0]:>14}   {cm[0, 1]:>15}")   
print(f"Actual Home Win  {cm[1, 0]:>14}   {cm[1, 1]:>15}")   



# visual heatmap of confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Away Win', 'Predicted Home Win'], # Corrected x-axis labels
            yticklabels=['Actual Away Win', 'Actual Home Win'])   # Corrected y-axis labels
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()


# Find out which features are most important for predictions

feature_importance = pd.DataFrame({
    'feature': feature_columns_for_model,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
}).sort_values('abs_coefficient',ascending=False)

print("\nFeature Importance (Logistic Regression Coefficients:)")
print(feature_importance)

# Explain what the model learned in plain English
# ...existing code...
print("\nModel Interpretation:")
for idx, row in feature_importance.iterrows():
    direction = "increases" if row['coefficient'] > 0 else "decreases"
    impact = "strong" if row['abs_coefficient'] > 0.1 else "moderate" if row['abs_coefficient'] > 0.05 else "weak"
    print(f"- {row['feature']}: {direction} home team win probability ({impact} impact)")



