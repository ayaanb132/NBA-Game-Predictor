# HomeCourt AI 🏀

This is a personal side project for my resume — a fun way to learn more about machine learning and sports data.

## What’s the goal?

This project uses NBA data from an API, pulls stats about teams and players, and feeds it all into a machine learning model. The big idea is to train the model on past games and teach it to predict which team will win future home games. 

It’s a work in progress, so expect changes and improvements as I learn new stuff!

## How does it work?

- **Grab the data:** Connects to an NBA stats API to fetch real game data.
- **Preps the info:** Turns raw stats into features the model can understand (like home/away records, player stats, etc).
- **Trains the model:** Uses those features to teach a machine learning model to spot patterns and make predictions.
- **Makes predictions:** Tries to guess the winner for upcoming home games.

## Why am I doing this?

- To practice working with APIs and data wrangling  
- To learn about model training and evaluation  
- To build something cool for my portfolio

## Still to do

- Add more features (injury reports, win streaks, etc)
- Try different types of models
- Make a simple UI or dashboard for predictions
- Clean up the code and docs!

## Getting Started

1. Clone the repo
2. Install dependencies (`pip install -r requirements.txt`)
3. Get an NBA API key and add it to a `.env` file
4. Run with `python main.py`

---

**Not affiliated with the NBA — just for learning and fun!**
