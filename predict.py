import joblib
import pandas as pd
from config import MODEL_PATH

def predict_game(game_features: dict):
    model = joblib.load(MODEL_PATH)

    data = pd.DataFrame([game_features])

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data).max()

    result = "Home Team Wins" if prediction == 1 else "Home Team Loses"

    return result, float(probability)


if __name__ == "__main__":
    sample_game = {
        "FG_PCT_HOME": 0.48,
        "FG_PCT_AWAY": 0.44,
        "REB_HOME": 45,
        "REB_AWAY": 41,
        "AST_HOME": 26,
        "AST_AWAY": 22,
        "TURNOVERS_HOME": 11,
        "TURNOVERS_AWAY": 14
    }

    result, confidence = predict_game(sample_game)

    print("Prediction:", result)
    print(f"Confidence: {confidence:.2%}")
