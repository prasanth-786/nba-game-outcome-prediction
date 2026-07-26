from sklearn.ensemble import RandomForestClassifier

def build_model(random_state=42):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state
    )
    return model
