# NIDS - Network Intrusion Detection System

Deep Learning-based NIDS using the UNSW-NB15 dataset.

## Dataset

- **Source:** UNSW-NB15 (CSV)
- **Training:** `dataset/UNSW_NB15_training-set.csv` (175,341 records)
- **Testing:** `dataset/UNSW_NB15_testing-set.csv` (82,332 records)
- **Task:** Binary classification (0 = Normal, 1 = Attack)

## Pipeline

1. **Cleaning** - Replace `-` in `service` column with NaN, drop missing rows
2. **Encoding** - One-Hot Encoding on `proto`, `service`, `state`
3. **Scaling** - MinMaxScaler to [0, 1]
4. **Balancing** - SMOTE on training set (1:1 ratio)
5. **Feature Selection** - Extra Trees Classifier, Top 8 features
6. **Model** - Sequential DNN (800-800-400), ReLU, Dropout 0.2, L2 reg, Softmax
7. **Training** - AdamW (lr=0.001, wd=0.01), categorical_crossentropy, 100 epochs, batch 50, val split 33%
8. **Evaluation** - Accuracy, Precision, Recall, F1, AUC-ROC, Confusion Matrix

## Project Structure

```
nids-deep-learning/
├── dataset/
│   ├── UNSW_NB15_training-set.csv
│   └── UNSW_NB15_testing-set.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── models/
├── reports/
├── requirements.txt
├── main.py
└── README.md
```

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Requirements

- Python 3.10
- TensorFlow 2.10
- CUDA 11.2 / cuDNN 8.1 (optional, for GPU)

## Expected Results

| Metric   | Target   |
|----------|----------|
| Accuracy | ~97.93%  |
| AUC-ROC  | ~0.99    |
