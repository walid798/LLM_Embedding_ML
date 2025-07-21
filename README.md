# LLM_Embedding_ML Hybrid Classifier

This repository demonstrates how to integrate large language model (LLM) embeddings with traditional machine learning models using Scikit-learn. It showcases a feature engineering pipeline that combines semantically rich text embeddings with structured data to train classifiers on text-heavy tasks.

## Models Compared

We evaluated multiple SentenceTransformer models on a customer support classification dataset. Each model's embeddings were combined with structured features and passed to a Random Forest classifier.


## Project Structure

```
llm-embedding-scikit/
│
├── data/
│   └── customer_support_dataset.csv      # Dataset (optional local copy)
│
├── notebooks/
│   └── 01_llm_model_comparison.ipynb     # Main notebook (multi-model benchmarking + t-SNE)
│
├── src/
│   └── embedding_pipeline.py             # Reusable feature engineering and embedding functions
│   └── model_utils.py                    # Training, evaluation, and visualization utilities
│
├── README.md                             # Project overview, models compared, usage guide
├── requirements.txt                      # All dependencies (sentence-transformers, sklearn, etc.)
└── LICENSE
```

## Visualization

We include a t-SNE 2D visualization of the sentence embeddings from each model, color-coded by class label.

## Getting Started

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Then run the notebook in the `notebooks/` folder.

## License

MIT License.
