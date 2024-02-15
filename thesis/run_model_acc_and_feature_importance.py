import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datasets import ADULT, Lawschool, INSURANCE
import category_encoders as ce

def filter_and_convert_keys(input_dict):
    filtered_dict = {key: value for key, value in input_dict.items() if value is not None}
    keys_list = list(filtered_dict.keys())
    return keys_list



# Load a dataset
dataset = Lawschool()
binary = False

features = dataset.features
label = dataset.label

cat_columns = filter_and_convert_keys(features)
cat_columns = list(filter(lambda item: item is not label, cat_columns))
train_data_df = dataset.raw

X = train_data_df.drop(label, axis=1)  # Features
y = train_data_df[label]  # Target variable

if(binary):
    # Create BinaryEncoder object
    encoder = ce.BinaryEncoder(cols=cat_columns)

    # Fit and transform the data
    X = encoder.fit_transform(X)
else:
    # Convert categorical features to numerical using one-hot encoding
    X = pd.get_dummies(X, columns=cat_columns)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

print(f'============={dataset.name} ({"Binary" if binary else "One-hot"})===============')

print("Number of dimensions/features after encoding:", X.shape)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

# Display classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Display confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Get feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("\nFeature ranking:")
for f in range(10):
    print(f"{f + 1}. {X_train.columns[indices[f]]} ({importances[indices[f]]:.3f})")
