import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from config import DATASET_PATH, MODEL_PATH, TEST_SIZE, RANDOM_STATE
from preprocess import load_data
from model import build_model

X, y = load_data(DATASET_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

model = build_model(RANDOM_STATE)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2%}")
print("\nClassification Report")
print(classification_report(y_test, predictions))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")
