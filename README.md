# Customer churn prediction

## Problem Statement
I aimed to predict customer churn for a telecommunications company based on customer attributes and usage patterns.

## Approach
My approach to solving this problem involved the following steps:

### Data Preprocessing:

Column Selection: I identified 'Gender' and 'Location' as categorical columns, while assuming the rest are numerical features.
Preprocessing Transformer: I created a preprocessing transformer using ColumnTransformer. Numerical columns were standardized using StandardScaler, and categorical columns were one-hot encoded using OneHotEncoder with settings to handle unknown categories.
Feature Transformation: The preprocessing transformer was applied to the entire dataset, resulting in X_preprocessed. This transformed data was then converted into a sparse matrix, X_sparse, using csr_matrix.
### Model Selection:

Feature Selection: For feature selection, I used a RandomForestClassifier in combination with SelectFromModel. This allowed me to select important features based on feature importances calculated by the RandomForestClassifier, with a maximum depth of 4 for the decision trees.
I also tried other models :
* LinearSVC
* LogisticRegression
* XGBoost
* Neural Networks
Classification Model: After feature selection, I built a classification pipeline that included the feature selector and another RandomForestClassifier for classification. The classifier was configured with parallel processing (n_jobs=-1) and a maximum depth of 4.
### Model Training and Evaluation:

Model Training: The entire pipeline, which encompassed data preprocessing, feature selection, and classification, was fitted to the training data (X_sparse) and labels (y).
Model Evaluation: I made predictions (y_pred) on the same training data and calculated the following evaluation metrics:
Train Accuracy: The accuracy of the model on the training data.
Confusion Matrix: A table summarizing true positives, true negatives, false positives, and false negatives.
Classification Report: A detailed report including precision, recall, F1-score, and support for each class.
## Results
The model achieved a training accuracy of 0.51, as well as a detailed breakdown of performance in the classification report. 

