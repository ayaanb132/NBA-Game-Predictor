import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# load main datasets

player_stats = pd.read_csv('/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data/Player Per Game.csv')
team_stats = pd.read_csv('/Users/ayaanbaig/Desktop/HomeCourt_AI/nba_data/Team Totals.csv')
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

print("Team Stats Columns:")
print(team_stats.columns.tolist())

#checking for game-specific data 

print("\nUnique values in key columns:")
if 'season' in team_stats.columns:
    print("Seasons available:", team_stats['season'].nunique())
    print("Season range:",team_stats['season'].min(), "to", team_stats['season'].max())
if 'team' in team_stats.columns:
    print("Teams:", team_stats['team'].unique()[:10]) # shows first 10 teams
