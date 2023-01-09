# Bank Marketing

Data Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Metadata: data/bank-names.txt

---

## Target Count
![Imbalanced target!](/figure/number_of_by_y.png)

# Ooops..!! Imbalanced target. Let's Use SMOTE!
![Balanced Target!](/figure/y_after_smote.png)

# Using Logistic Regression
|fold|   accuracy|  precision| recall  | f1     |
|----|-----------|-----------|---------|-------|
|0 | 0.802068   |0.978389  |0.617781|  0.757350|
|1  |0.917229   |0.937049  |0.894555 | 0.915309|
|2  |0.881185   |0.859483  |0.911371 | 0.884667|
|3  |0.872010   |0.848238  |0.906121 | 0.876225|
|4  |0.848852   |0.812855  |0.906409 | 0.857087|
|----|-----------|-----------|---------|-------|
|avg.|0.864269|0.887203|0.847248|0.858128|

# Using Decision Tree Classifier
|fold |  accuracy | precision  |  recall   |     f1|
|------|---------|-----------|---------|--------|
|0  |0.751413   |0.751586  |0.751068 | 0.751327|
|1  |0.810889   |0.733755  |0.975879 | 0.837672|
|2  |0.806754   |0.728186  |0.978911 | 0.835136|
|3  |0.775243   |0.695142  |0.980425 | 0.813497|
|4  |0.625198   |0.572907  |0.984011 | 0.724183|
|----|-----------|-----------|---------|-------|
|avg.|0.753899|0.696315|0.934059|0.792363|