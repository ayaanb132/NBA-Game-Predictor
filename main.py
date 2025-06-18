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

'''
LOADING DATASET AND MERGING 2 CSV FILES
'''
player_stats = pd.read_csv('/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data/Player Per Game.csv')

team_summaries = pd.read_csv('/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data/Team Summaries.csv')
team_totals = pd.read_csv('/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data/Team Totals.csv')
team_totals_subset = team_totals[['season', 'team', 'lg', 'g', 'pts', 'fg_percent']]

# merging summaries with totals: 
team_stats = pd.merge(
    team_summaries,
    team_totals_subset,
    on=['season','team','lg'],
    how='inner'
)

player_totals = pd.read_csv('/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data/Player Totals.csv')

#display basic information about the datasets using .shape

print("Player Stats Shape:", player_stats.shape)
print("Team Stats Shape:", team_stats.shape)
print("Player Totals Shape:", player_totals.shape)

#examine first few rows of team stats

# .head() displays first 5 rows
print("\nTeam Stats - First 5 rows:")
print(team_stats.head())

#.info() gives data types and missing value counts
print("\nTeam Stats - Column Information:")
print(team_stats.info())

#.describe() gives statistical summaries of numerical columns
print("\nTeam Stat - Basic Statistics")
print(team_stats.describe())






'''
PREPARING GAME-LEVEL DATA
'''

# examining what columns are in team_stats
print("Team Stats Columns:")
print(team_stats.columns.tolist())

#checking for game-specific data 

print("\nUnique values in key columns:")
if 'season' in team_stats.columns:
    print("Seasons available:", team_stats['season'].nunique())
    print("Season range:",team_stats['season'].min(), "to", team_stats['season'].max())
if 'team' in team_stats.columns:
    print("Teams:", team_stats['team'].unique()[:10]) # shows first 10 teams

team_season_stats =team_stats.copy()
print("\nWorking with team season statistics")
print("Sample of team season data:")
print(team_season_stats.head())

'''
DATA CLEANING
'''

# check for missing values:

print("Missing values per column:")
missing_values = team_season_stats.isnull().sum()
print(missing_values[missing_values > 0])

#handle missing values
#used median imputation to fill NaN values

numerical_columns = team_season_stats.select_dtypes(include=[np.number]).columns

for col in numerical_columns:
    if team_season_stats[col].isnull().sum() > 0:
        median_value = team_season_stats[col].median()
        team_season_stats[col].fillna(median_value, inplace=True)
        print(f"Filled {col} missing values with median: {median_value}")

#check data types and convert if necessary
print("\nData types:")
print(team_season_stats.dtypes)

#clean team names (removing extra characters and white space)

if 'team' in team_season_stats.columns:
    team_season_stats['team'] = team_season_stats['team'].str.strip()


columns_to_remove = []

#remove irrelevant or leakage columns
manual_remove = ['lg','abbreviation','playoffs','arena','attend','attend_g']

#adding generic unwanted columns
for col in team_season_stats.columns:
    if col.lower() in ['unnamed','index','rk'] or 'unnamed' in col.lower():
        columns_to_remove.append(col)
columns_to_remove.extend(manual_remove)

team_season_stats.drop(columns=columns_to_remove, inplace=True)

print(f"\nCleaned dataset shape: {team_season_stats.shape}")

'''
CREATING SIMULATED MATCHUPS AND OUTCOMES FOR MODEL
'''



def create_game_data(season_stats, n_games_per_season=50):
    '''
    Create simulated game data from season statistics
    '''
    games = []

    # get unique teams and seasons

    teams = season_stats['team'].unique()
    seasons = season_stats['season'].unique() if 'season' in season_stats.columns else [2025]

    for season in seasons:
        season_data = season_stats[season_stats['season'] == season] if 'season' in season_stats.columns else season_stats

        # create random matchups
        for i in range(n_games_per_season):
            if len(season_data) < 2:
                continue
    
            #randomly select home and away teams
            home_away_teams = np.random.choice(season_data['team'].values,2,replace=False)
            home_team = home_away_teams[0]
            away_team = home_away_teams[1]

            # get stats for both teams
            home_stats = season_data[season_data['team'] == home_team].iloc[0]
            away_stats = season_data[season_data['team']== away_team].iloc[0]

            game_record = {
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'home_wins': home_stats.get('w',0), # grabs wins or places 0 if win not there
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


# create game dataset
print("creating game-level data from season statistics...")
games_df = create_game_data(team_season_stats,n_games_per_season=100)
print(f"Created {len(games_df)} game records")
print("\nSample game data:")
print(games_df.head())

'''
FEATURE ENGINEERING - creating features like win percentage and relatinve team strength
'''

def engineer_features(games_df):
    """
    create features for predicting home team wins
    """

    df = games_df.copy()


    #calculate win percentages

    df['home_win_pct'] = df['home_wins'] / (df['home_wins'] + df['home_losses'])
    df['away_win_pct'] = df['away_wins'] / (df['away_wins'] + df['away_losses'])

    ## fill NaN values with 0.5 (neutral win rate)
    df['home_win_pct'] = df['home_win_pct'].fillna(0.5) 
    df['away_win_pct'] = df['away_win_pct'].fillna(0.5)

    # differential features (home team advantage)

    df['win_pct_diff'] = df['home_win_pct'] - df['away_win_pct']
    df['ppg_diff'] = df['home_ppg'] - df['away_ppg']
    df['fg_pct_diff'] = df['home_fg_pct'] - df['away_fg_pct']

    # home court advantage
    df['home_court_advantage'] = 1

    # team strength indicators
    df['home_team_strong'] = (df['home_win_pct']>0.55).astype(int)
    df['away_team_strong'] = (df['away_win_pct']>0.55).astype(int)

    return df


print("Engineering features....")
games_df = engineer_features(games_df)


# dispkay the engineered features

print("Engineered features:")
feature_columns = ['home_win_pct', 'away_win_pct', 'win_pct_diff','ppg_diff','fg_pct_diff','home_court_advantage','home_team_strong','away_team_strong']
print(games_df[feature_columns].head())

print("\nFeature statistics:")
print(games_df[feature_columns].describe())


def create_target_variable(df):
    """
    creating binary target variable for home team wins using probabilistic
    approach based on team strength
    """

    # creating a probability based on features

    home_win_prob = (
        0.50 + # base home court advantage at 50%
        0.30 * df['win_pct_diff'] + # win percentage difference impact
        0.15 * (df['ppg_diff'] /20) + # ppg difference (normalized by dividing by 20)
        0.05 * df['fg_pct_diff'] # field goal percentage difference
    )

    # making sure probabilites are between 10% and 90% (no impossible probabilites)

    home_win_prob = np.clip(home_win_prob,0.1,0.9)

    # make binary outcomes based on the probability

    np.random.seed(42) # to make results reproducible
    df['home_team_win'] = np.random.binomial(1, home_win_prob)

    return df

# target variable

print("Creating target varibale...")
games_df = create_target_variable(games_df)

# check the distribution of outcomes

print("Home team win distribution:")
print(games_df['home_team_win'].value_counts())
print(f"Home team win rate: {games_df['home_team_win'].mean() * 100:.3f}%")

correlation_with_target = games_df[feature_columns + ['home_team_win']].corr()['home_team_win'].sort_values(ascending=False)
print("\nCorrelation with home team wins:")
print(correlation_with_target)


'''
MODEL BUILDING
'''

# prepare the features for modeling

feature_columns_for_model = ['home_win_pct', 'away_win_pct', 'win_pct_diff','ppg_diff','fg_pct_diff']

X = games_df[feature_columns_for_model]
y = games_df['home_team_win']

print("Features for modeling:")
print(X.head())
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# splitting data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} games")
print(f"Testing set size: {X_test.shape[0]} games")
print(f"Training set home win rate: {y_train.mean():.3f}")
print(f"Testing set home win rate: {y_test.mean():.3f}")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature scaling completed")
print("Training features mean after scaling: ", X_train_scaled.mean(axis=0).round(3))
print("Training features std after scaling: ", X_train_scaled.std(axis=0).round(3))

# creating and training logistic regression model

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled,y_train)

print("\nModel training completed!")

# make prediction on the test set

y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:,1] # probability of home team winning

print("Predicitons generated")
print(f"Predicted home wins: {y_pred.sum()}")
print(f"Actual home wins: {y_test.sum()}")

'''
MODEL EVAULATION
'''

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Away Win', 'Home Win']))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix")
print("Predicted:  Away Win  Home Win")
print(f"Away Win:      {cm[0,0]:3d}      {cm[0,1]:3d}")
print(f"Home Win:      {cm[1,0]:3d}      {cm[1,1]:3d}")

# Visual confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Away Win', 'Home Win'], yticklabels=['Away Win', 'Home Win'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()


# analyze feature importance

feature_importance = pd.DataFrame({
    'feature': feature_columns_for_model,
    'coefficient': model.coef_[0],
    'abs_coefficient': np.abs(model.coef_[0])
}).sort_values('abs_coefficient',ascending=False)

print("\nFeature Importance (Logistic Regression Coefficients:)")
print(feature_importance)

# interpret results
print("\nModel Interpretation:")
for idx, row in feature_importance.head(3).iterrows():
    direction = "increases" if row['coefficient'] > 0 else "decreases"
    print(f"- {row['feature']}: {direction} home team win probability")



