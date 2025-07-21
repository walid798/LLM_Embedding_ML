
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

def load_data(url: str) -> Tuple[List[str], np.ndarray, List[str]]:
    df = pd.read_csv(url)
    text_data = df["text"].tolist()
    structured_data = df[["prior_tickets", "account_age_days"]].values
    labels = df["label"].tolist()
    return text_data, structured_data, labels

def get_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True)

def scale_features(numerical_data: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(numerical_data)

def combine_features(structured_scaled: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    return np.hstack([structured_scaled, embeddings])
