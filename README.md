# NBA Game Outcome Prediction

## Overview

This project predicts whether the **home NBA team will win or lose** using historical game statistics and a **Random Forest Classifier**.

The project demonstrates a complete machine learning workflow:
- Data preprocessing
- Model training
- Model evaluation
- Prediction on new game statistics

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook

---

## Project Structure

```
nba-game-outcome-prediction/
├── config.py
├── preprocess.py
├── model.py
├── train.py
├── predict.py
├── utils.py
├── requirements.txt
├── LICENSE
├── README.md
├── dataset/
├── models/
├── docs/
├── images/
└── results/
```

---

## Dataset

Place the NBA dataset (`nba_games.csv`) inside the `dataset/` folder.

The target column is:

```
HOME_TEAM_WINS
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Running the Project

1. Add the dataset to the `dataset/` folder.
2. Run:

```bash
python train.py
```

3. After training, make predictions using:

```bash
python predict.py
```

---

## Machine Learning Model

- Random Forest Classifier
- 100 Decision Trees
- Train/Test Split
- Accuracy and Classification Report

---

## Future Improvements

- Hyperparameter tuning
- Feature engineering
- XGBoost and LightGBM comparison
- Interactive web application using Streamlit

---

## Author

Alla Prasanth
