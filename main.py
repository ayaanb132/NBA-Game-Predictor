import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

'''
LOADING DATASET AND MERGING 2 CSV FILES
'''
player_stats = pd.read_csv('/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data/Player Per Game.csv')

team_summaries = pd.read_csv('/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data/Team Summaries.csv')
team_totals = pd.read_csv('/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data/Team Totals.csv')
team_totals_subset = team_totals[['season', 'team', 'lg', 'g', 'pts']]

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

# examining what columns are in team_stats




'''
PREPARING GAME-LEVEL DATA
'''


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


