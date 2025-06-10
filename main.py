import requests
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from dotenv import load_dotenv
import os

load_dotenv


API_KEY = os.getenv("API_KEY")


BASE_URL = "https://api.balldontlie.io/v1/games" # URL for nba games API
SEASON = 2023 # season year

all_games = [] # empty list to store all game data

headers = {
    "Authorization": API_KEY  # Pass API key in headers
}
cursor = None

def fetch_with_retry(url, params,headers,max_retries = 5, initial_wait = 2):
    """
    Tries to fetch data from an API endpoint while exponentially backing off, avoiding a 
    429 error

    Arguments:
        url (str): The API URL to call
        headers (dict): Request headers (e.g., Authorization)
        params (dict): Query parameters (e.g., cursor, season)
        max_retries (int): Max number of retry attempts
        initial_wait (float): Initial delay in seconds

        Returns:
        dict or None: The JSON response if successful, else None
    """
    
    retries = 0
    while retries < max_retries:
        try:
            # GET request
            response = requests.get(url, params=params, headers=headers)

            # handle rate limiting if 429 error occurs
            if response.status_code == 429:
                wait_time = initial_wait * (2 ** retries)
                print(f"Rate limit reached, retrying in {wait_time:.1f} seconds")
                time.sleep(wait_time)
                retries+=1
                continue

            response.raise_for_status() # Raising HTTP error for bad response

    
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            retries +=1  
            if retries > max_retries:
                print("Max retries reached. Giving up.")
                return None
            wait_time = initial_wait * (2 ** retries)
            print(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
        except requests.exceptions.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            return None

    print("Max retries reached. Giving up")
    return None



# GET REQUEST FOR THE DATA
while True:
    params = {
        "seasons[]": SEASON
    }

    if cursor:
        params["cursor"] = cursor
    try:
        data = fetch_with_retry(url = BASE_URL, params=params, headers=headers)
        if not data:
            print("No data returned")
            break
        

        all_games.extend(data["data"])

        # check for next cursor to continue pagination
        cursor = data.get("meta", {}).get("next_cursor")
        
        if not cursor:
            break # no more pages
        time.sleep(1)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        break

    

print(f"Total games fetched: {len(all_games)}")

# USING NUMPY TO CONVERT JSON TO DATAFRAME
df = pd.DataFrame(all_games) # convert the list of games into a dataframe
print()
print("FIRST COUPLE ROWS: \n")
print(df.head()) # look at first few rows
print("CHECK FOR MISSING ROWS: \n")
print()
print(df.info()) # check for missing values
print("CHECK NUMBER OF ROWS AND COLUMNS: \n")
print()
print(df.shape) # check number of rows and columns
print("SUMMARY STATISTICS: \n")
print()
print(df.describe()) # summary statistics for numeric columns
print()
print("\n")

# show number of missing values in each columns:
print("NUMBER OF MISSING VALUES IN EACH COLUMNS: ")
print(df.isnull().sum())

# remove the rows that have missing values: 
df.dropna(inplace=True)

# Calcualte point difference between teams
df['PointDifference'] = df['home_team_score'] - df['visitor_team_score']


# create a label column: 1 if home team wins, else 0
df['HomeWin'] = (df['PointDifference'] > 0).astype(int)

# X = input features, y - output labels

X = df[['PointDifference']] # features
y = df['HomeWin'] # target

# 80% training, 20% testing

X_train, X_test, y_train ,y_test= train_test_split(X,y,test_size=0.2,random_state=42)
print()
print()

model = LogisticRegression() # Create a logicitc regression model
print("Training data....")
model.fit(X_train, y_train) # train it on the training data

y_pred = model.predict(X_test) # predict on the test set

print()
print()
# check overall accuracy
print("Accuracy: ", accuracy_score(y_test, y_pred))

# show where the model was right and wrong

cm = confusion_matrix(y_test,y_pred)
cm_df = pd.DataFrame(cm, 
                    index = ['Actual 0 ', 'Actual 1'],
                    columns=['Predicted 0 ', 'Predicted 1'])
print("Confusion Matrix:\n",cm_df)








