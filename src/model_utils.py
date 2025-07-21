
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.manifold import TSNE
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict

def train_classifier(X_train: np.ndarray, y_train: List[str]) -> RandomForestClassifier:
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf: RandomForestClassifier, X_test: np.ndarray, y_test: List[str]) -> Dict[str, float]:
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 (Weighted)": f1_score(y_test, y_pred, average="weighted")
    }

def visualize_tsne(embeddings: np.ndarray, labels: List[str], title: str = "t-SNE Visualization"):
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    df = pd.DataFrame(reduced, columns=["x", "y"])
    df["label"] = labels

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=df, x="x", y="y", hue="label", palette="tab10", s=100)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
