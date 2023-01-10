# Bank Marketing

Data Source: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing

Metadata: data/bank-names.txt

---

# Target Count
![Imbalanced target!](/figure/number_of_by_y.png)

# Ooops..!! Imbalanced target. Let's Use SMOTE!
![Balanced Target!](/figure/number_of_by_y_after_SMOTE.png)

# Logistic Regression, Decision Tree, Random Forest, Decision Tree, Gaussian Naive Bayes

Logistic Regression : accuracy : 0.88 <br />
Logistic Regression : precision : 0.9 <br />
Logistic Regression : recall : 0.86 <br />
Logistic Regression : f1 score : 0.88 <br />
<br />
Decision Tree Classifier : accuracy : 0.92 <br />
Decision Tree Classifier : precision : 0.91 <br />
Decision Tree Classifier : recall : 0.95 <br />
Decision Tree Classifier : f1 score : 0.93 <br />
<br />
Random Forest Classifier : accuracy : 0.95 <br />
Random Forest Classifier : precision : 0.94 <br />
Random Forest Classifier : recall : 0.96 <br />
Random Forest Classifier : f1 score : 0.95 <br />
<br />
Gaussian Naive Bayes : accuracy : 0.75 <br />
Gaussian Naive Bayes : precision : 0.71 <br />
Gaussian Naive Bayes : recall : 0.84 <br />
Gaussian Naive Bayes : f1 score : 0.77 <br />
<br />


# Probabilistic Algorithm (Logistic Regression, Random Forest, Gaussian Naive Bayes)

## ROC Curve

![ROC!](/figure/ROC_AUC_Score.png)


## Good Work! Random Forest!!
---

# Cross Validation with 10 Fold
<br />
### Metrics: accuracy <br />
[0.76333563 0.92708477 0.9371468  0.92212267 0.93301172 <br />
 0.89000689 0.86037216 0.85265334 0.85277089 0.76936862] <br />
<br />
### Metrics: precision <br />
[0.92473592 0.90769231 0.89641434 0.87315062 0.89375467  <br />
 0.81148649 0.7838485  0.77074423 0.77500538 0.68132905] <br />
<br />
### Metrics: recall <br />
[0.57375241 0.96057348 0.99338296 0.99283154 0.99173098 <br />
 0.99255788 0.99255788 0.99393605 0.9917287  0.99476151] <br />
<br />
### Metrics: f1 <br />
[0.70435525 0.93112653 0.93844549 0.9294938  0.9372636  <br />
 0.89004201 0.88138909 0.86974334 0.86632948 0.81084121] <br />