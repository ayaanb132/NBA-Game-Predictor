# 🏀 HomeCourt AI – Predicting NBA Home Team Wins with Machine Learning

**HomeCourt AI** is an end-to-end machine learning project that predicts whether the home team will win a given NBA game using over 75 years of professional basketball data. It showcases skills in data engineering, feature selection, modeling, and (soon) deployment — all tied together in a real-world sports analytics application.

> ⚡ **Tech Stack**: Python, Pandas, Scikit-learn, Jupyter, NumPy\
> 📊 **ML Models**: Logistic Regression, Decision Trees *(XGBoost planned)*\
> 🚀 **Next Steps**: Model API, Streamlit dashboard, live data integration

---

## 📌 Project Highlights

✅ **Complete ML Pipeline**: Built end-to-end logistic regression model with 74%+ accuracy\
✅ **Advanced Feature Engineering**: Win percentage differentials, PPG gaps, FG% comparisons\
✅ **Smart Data Simulation**: Created 100+ game scenarios per season from team statistics\
✅ **Robust Data Cleaning**: Automated removal of leakage columns and missing value imputation\
✅ **Comprehensive Evaluation**: Confusion matrices, feature importance analysis, and visual heatmaps\
✅ **Professional Codebase**: Modular functions, proper scaling, and reproducible results\
✅ **Multi-Dataset Integration**: Merged team summaries, totals, and player statistics seamlessly

---

## 📁 Folder Structure

```
HomeCourt-AI/
├── nba_data/              # Raw NBA stats (games, players, teams)
├── main.py                # Main training & prediction pipeline
├── jupyter_converter.py   # Converts CSVs into notebook-friendly format
├── csv_viewer.ipynb       # Explore datasets and debug issues
├── requirements.txt       # Python dependencies
└── README.md              # You're here!
```

---

## 🔮 Prediction Objective

> Given a matchup, predict whether the **home team** wins (`1`) or loses (`0`).

### Features Engineered:

- **Win Percentage Differentials**: Home vs away team win rates
- **Performance Gaps**: Points per game and field goal percentage differences  
- **Team Strength Indicators**: Binary flags for teams with >55% win rate
- **Home Court Advantage**: Built-in 1.0 coefficient for home team bias
- **Probabilistic Target Creation**: Realistic win probabilities based on team strength
- *(Coming soon)* Momentum, injuries, and team fatigue factors

---

## 📈 Model Performance (Latest Results)

| Model               | Accuracy | Precision | Recall | Notes                              |
| ------------------- | -------- | --------- | ------ | ---------------------------------- |
| Logistic Regression | ~74.5%   | 0.75      | 0.73   | **Current baseline** - Well-balanced |
| Decision Tree       | ~68.2%   | 0.71      | 0.65   | Overfitting on small samples       |
| Random Forest       | Coming   | -         | -      | Ensemble method in development      |

### Key Performance Insights:
- **Home Court Advantage**: Model correctly identifies ~6-8% home team bias
- **Feature Importance**: Win percentage differential is the strongest predictor
- **Balanced Predictions**: Minimal bias between home/away predictions
- **Stable Results**: Consistent performance across multiple random seeds

> 🔧 Model uses StandardScaler normalization and stratified train/test splits for reliable evaluation.

---


## 🧠 Recent Updates & Roadmap

### ✅ **Latest Commits (June 2025)**:
- **Complete ML Pipeline**: Implemented full logistic regression model with feature scaling
- **Advanced Feature Engineering**: Win%, PPG, and FG% differentials with home court advantage
- **Smart Game Simulation**: Generate realistic matchups from season statistics (100 games/season)
- **Automated Data Cleaning**: Intelligent removal of text columns and data leakage features
- **Comprehensive Evaluation**: Confusion matrices, accuracy metrics, and feature importance analysis
- **Visual Analytics**: Seaborn heatmaps for confusion matrix and model performance visualization
- **Modular Architecture**: Separated data loading, cleaning, feature engineering, and evaluation

### 🚀 **Next Sprint**:
- **Model Optimization**: Hyperparameter tuning and cross-validation
- **Feature Expansion**: Player injury data, team momentum, travel fatigue
- **API Development**: FastAPI endpoint for real-time predictions
- **Dashboard Creation**: Streamlit interface with interactive visualizations
- **Live Integration**: Real-time NBA data feeds for current season predictions

---

## 🛠 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/ayaanb132/HomeCourt-AI.git
cd HomeCourt-AI

# Set up virtual environment (recommended)
python3 -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the complete ML pipeline
python main.py

# Optional: Explore datasets interactively
jupyter notebook csv_viewer.ipynb
```

### 🎯 **What You'll See:**
- Dataset loading and merging statistics
- Feature engineering and correlation analysis  
- Model training progress and final accuracy
- Confusion matrix heatmap visualization
- Feature importance rankings with interpretations

---

## 📬 Contact & Contribution

Feel free to reach out or open an issue if you want to contribute ideas, models, or datasets.

📧 [ayaanb132@gmail.com](mailto\:ayaanb132@gmail.com)\
📍 Built by Ayaan B. – *Computer Science @ Wilfrid Laurier University*

