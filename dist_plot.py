import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Read the CSV file
true_df = pd.read_csv('csv/4_16_ground_truth.csv')
rec_df = pd.read_csv('csv/4_16_reconstructed.csv')

label = 'salary'

features = true_df.columns.tolist()
features.remove(label)

true_X = true_df.drop(label, axis=1)  # Features
true_y = true_df[label]  # Target variable

rec_X = rec_df.drop(label, axis=1)  # Features
rec_y = rec_df[label]  # Target variable


# print('True value count:')
# print(true_df['salary'].value_counts())

# print('Rec value count:')
# print(rec_df['salary'].value_counts())

# rec_X = pd.get_dummies(rec_X, columns=features)
# X_train, X_test, y_train, y_test = train_test_split(rec_X, rec_y, test_size=0.2, random_state=42)

# # Initialize and train the Random Forest model
# # model = RandomForestClassifier(n_estimators=100, random_state=42)
# # model.fit(X_train, y_train)
# # y_pred = model.predict(X_test)
# # conf_matrix = confusion_matrix(y_test, y_pred)
# # print("Confusion Matrix:")
# # print(conf_matrix)

# true_X = pd.get_dummies(true_X, columns=features)
# X_train_true, X_test_true, y_train_true, y_test_true = train_test_split(true_X, true_y, test_size=0.2, random_state=43)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
# y_pred_true  = model.predict(X_test_true)
# conf_matrix = confusion_matrix(y_test_true, y_pred_true)
# print("Confusion Matrix:")
# print(conf_matrix)


# Ground-truth:
# [[ 418  304]
#  [ 143 2053]]

# Reconstructed:
# [[ 310  412]
#  [ 113 2083]]

for column in true_X:
    plt.figure(figsize=(16, 6))

    # Calculate x-axis limits
    x_min = max(true_df[column].min(), rec_df[column].min())
    x_max = min(true_df[column].max(), rec_df[column].max())

    plt.subplot(1, 2, 1)
    sns.histplot(data=true_df, x=column, hue=label, kde=True)
    plt.title(f'Distribution of {column} (Truth)')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend(title=column)
    plt.xlim(x_min, x_max)  # Set x-axis limits

    plt.subplot(1, 2, 2)
    sns.histplot(data=rec_df, x=column, hue=label, kde=True)
    plt.title(f'Distribution of {column} (Reconstructed)')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend(title=column)
    plt.xlim(x_min, x_max)  # Set x-axis limits

    plt.tight_layout()
    plt.show()



# bucketing for the confusion matrix
# cat-boost