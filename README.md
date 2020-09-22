# FaultyProduct
### Overview

For this capstone project, we will use the SECOM manufacturing Data Set from the UCI Machine Learning Repository. The set is originally for semiconductor manufacturing, but in our case, we will assume that it is for the diaper manufacturing process.

**Data**
The dataset consists of two files:
- Dataset file SECOM: containing 1567 examples, each with 591 features, presented in a 1567 x 591 matrix
- Labels: file listing the classifications and date time stamp for each example

**Data Cleaning**
I prepared the manufacturing data for modeling by performing the following:
- Merged the feature and label datasets
- Split data into training/tuning/testing (70/15/15)
- Performed initial exploratory data analysis:
    - The label is highly imbalanced. Only a small percentage (6.6%) of the cases have defects.
    - All features are numeric.
    - Plotted scatter matrix for the features most correlated with the label. Most of those features have close to normal distribution. 
    - Some of the features appears to be correlated with each other (ex. feature 158 and 293). This may cause a colinearity problem when modeling. 
- Cleaned data by:
    - Removing features that have missing values in more than 30% of the dataset
    - Input mean for features with fewer missing values
    - Applied standard scalar to the features
- Addressed class imbalance problem using SMOTE
- Applied Wrapper forward selection to feature select:
    - Performance (using logistic regression and AOC curve as the error metric) started leveling off with the top 50 features
    
**Initial Model build**

I built the following models to detect faulty products:

1. Decision Tree Model
2. Ensemble Model: Random Forest
3. SVM Model

Below are the results (on the tune set):

| Metric | Decision Tree  | Random Forest  | SVC  |
|------|------|------|------|
|   Accuracy  | 0.8213| 0.9191| 0.8213|
|   AUC  | 0.748| 0.756| 0.587|
|   Recall  | 0.5833| 0.25| 0.3333|
|   Precision  | 0.1590| 0.23| 0.1053|
|   F1 Score  | 0.25| 0.24| 0.16|

**Random Forest model** (entropy with max_depth of 15, max_features of 0.2, min_samples_leaf of 5, and 50 n_estimators) performed the best in terms of both accuracy and F1 score. 

However it did not generalize as well to the test set. The model prediction on the test set had an accuracy score of 0.8729 and F1 score of 0.1667.

**Neural Network Models**

I built the following neural network models to detect faulty products:
1. Simple neural network
2. DNN model

Below are the results (on the tune set):

| Metric | Simple NN  | DNN  |
|------|------|------|
|   Accuracy  | 0.90638| 0.90638|
|   AUC  | 0.675| 0.689|
|   Recall  | 0.1667| 0.0833|
|   Precision  | 0.1428| 0.0833|
|   F1 Score  | 0.1538| 0.0833|

While the DNN did slightly better in terms of AUC, Random Forest (from Milestone 2) is still the better option with accuracy of 0.9191 and AUC of 0.756 on the tune set. 
