import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_

    plt.figure(figsize=(10,5))
    plt.bar(feature_names, importance)
    plt.xticks(rotation=45)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()
