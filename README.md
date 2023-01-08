# Bank Marketing

Data Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Metadata: data/bank-names.txt

---

## Target Count
![Imbalanced target!](/figure/number_of_by_y.png)

# Ooops..!! Imbalanced target. But, let just see the model evaluation.

Algorithm	: Decision Tree

Classification Report: 
|              | precision |   recall  | f1-score |  support |
|--------------|-----------|-----------|---------|------|
|           0  |     0.93   |   0.93   |   0.93   |   9029  |
|           1   |    0.44  |    0.43  |    0.44    |  1056|
|    accuracy  |           |        |      0.88   |  10085|
|   macro avg   |    0.69    |  0.68   |   0.68   |  10085|
|weighted avg   |    0.88   |   0.88   |   0.88   |  10085|