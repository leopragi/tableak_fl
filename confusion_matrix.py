import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix

def printPerformance(matrix):
    tn, fp, fn, tp = matrix.ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1_score = 2 * (precision * recall) / (precision + recall)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall (Sensitivity): {recall * 100:.2f}%")
    print(f"Specificity: {specificity * 100:.2f}%")
    print(f"F1 Score: {f1_score * 100:.2f}%")
    print(f"False Positive Rate: {fpr * 100:.2f}%")
    print(f"False Negative Rate: {fnr * 100:.2f}%")

# Read the CSV file
true_df = pd.read_csv('csv/2_4_ground_truth.csv')
rec_df = pd.read_csv('csv/2_4_reconstructed.csv')

label = 'salary'
cat_features = ['native-country', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'education', 'workclass']

features = true_df.columns.tolist()
features.remove(label)

true_X = true_df.drop(label, axis=1)  # Features
true_y = true_df[label]  # Target variable

rec_X = rec_df.drop(label, axis=1)  # Features
rec_y = rec_df[label]  # Target variable

# Preprocess the data
# rec_X = pd.get_dummies(rec_X, columns=features)
# true_X = pd.get_dummies(true_X, columns=features)

# Get common columns between X_train and X_test_true
common_columns = set(rec_X.columns).intersection(true_X.columns)

# Filter X_train and X_test_true to include only common columns
X_train = rec_X[list(common_columns)]
X_train_true = true_X[list(common_columns)]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, rec_y, test_size=0.2, random_state=42)
X_train_true, X_test_true, y_train_true, y_test_true = train_test_split(X_train_true, true_y, test_size=0.2, random_state=43)

model = CatBoostClassifier(random_state=41,verbose=False, cat_features=cat_features)
model.fit(X_train_true, y_train_true)

# Make predictions on the test set
y_pred_true = model.predict(X_test_true)

# Evaluate the predictions
conf_matrix = confusion_matrix(y_test_true, y_pred_true)
print("Original net Confusion Matrix:")
print(conf_matrix)
printPerformance(conf_matrix)

# Train the CatBoost model
model = CatBoostClassifier(random_state=42,verbose=False, cat_features=cat_features)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred_true = model.predict(X_test_true)

# Evaluate the predictions
conf_matrix = confusion_matrix(y_test_true, y_pred_true)
print("Recreated net Confusion Matrix:")
print(conf_matrix)
printPerformance(conf_matrix)

# Original net Confusion Matrix:
# [[ 885  401]
#  [ 229 3756]]
# Accuracy: 88.05%
# Precision: 90.35%
# Recall (Sensitivity): 94.25%
# Specificity: 68.82%
# F1 Score: 92.26%
# False Positive Rate: 31.18%
# False Negative Rate: 5.75%

# Reconstructed net Confusion Matrix:
# [[   1 1285]
#  [   9 3976]]
# Accuracy: 75.45%
# Precision: 75.57%
# Recall (Sensitivity): 99.77%
# Specificity: 0.08% (Never correctly any identifies negative cases, Good!)
# F1 Score: 86.00%
# False Positive Rate: 99.92% (Many mistakes classifying negative cases as positive, Good!)
# False Negative Rate: 0.23%


# Original net Confusion Matrix:
# [[ 443  260]
#  [ 147 2068]]
# Accuracy: 86.05%
# Precision: 88.83%
# Recall (Sensitivity): 93.36%
# Specificity: 63.02%
# F1 Score: 91.04%
# False Positive Rate: 36.98%
# False Negative Rate: 6.64%

# Recreated net Confusion Matrix:
# [[ 575  128]
#  [ 347 1868]]
# Accuracy: 83.72%
# Precision: 93.59%
# Recall (Sensitivity): 84.33%
# Specificity: 81.79%
# F1 Score: 88.72%
# False Positive Rate: 18.21%
# False Negative Rate: 15.67%