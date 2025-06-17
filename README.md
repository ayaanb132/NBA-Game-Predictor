# 🏀 HomeCourt AI – Predicting NBA Home Team Wins with Machine Learning

**HomeCourt AI** is an end-to-end machine learning project that predicts whether the home team will win a given NBA game using over 75 years of professional basketball data. It showcases skills in data engineering, feature selection, modeling, and (soon) deployment — all tied together in a real-world sports analytics application.

> ⚡ **Tech Stack**: Python, Pandas, Scikit-learn, Jupyter, NumPy\
> 📊 **ML Models**: Logistic Regression, Decision Trees *(XGBoost planned)*\
> 🚀 **Next Steps**: Model API, Streamlit dashboard, live data integration

---

## 📌 Project Highlights

✅ Built a binary classification model to predict home team wins\
✅ Engineered features from player, team, and game stats (1946–present)\
✅ Cleaned and merged multiple complex CSV datasets\
✅ Designed modular code for data prep, training, and evaluation\
✅ Visualized performance with accuracy metrics and test predictions\
✅ Created a custom Jupyter viewer for manual data inspection

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

### Features used:

- Team performance averages
- Player stats (PPG, RPG, AST, etc.)
- Game metadata (home/away, season, outcome)
- Opponent statistics
- *(Coming soon)* Momentum, injuries, and team fatigue factors

---

## 📈 Model Performance (Initial Results)

| Model               | Accuracy | Notes                        |
| ------------------- | -------- | ---------------------------- |
| Logistic Regression | \~XX%    | Baseline model               |
| Decision Tree       | \~XX%    | Overfitting on small samples |
| Random Forest       | Coming   |                              |

> 🔧 Future iterations will include model tuning, cross-validation, and ensembling for better generalization.

---

## 🧠 Roadmap

-

---

## 🛠 Setup Instructions

```bash
git clone https://github.com/ayaanb132/HomeCourt-AI.git
cd HomeCourt-AI
pip install -r requirements.txt
python main.py
```

You can also explore datasets using the `csv_viewer.ipynb` notebook.

---

## 📬 Contact & Contribution

Feel free to reach out or open an issue if you want to contribute ideas, models, or datasets.

📧 [ayaanb132@gmail.com](mailto\:ayaanb132@gmail.com)\
📍 Built by Ayaan B. – *Computer Science @ Wilfrid Laurier University*

