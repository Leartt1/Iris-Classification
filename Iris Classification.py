# Importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
dataset = pd.read_csv("iris.data.csv")

# Separate the features and the target variable
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Encode the target variable
le = LabelEncoder()
Y = le.fit_transform(Y)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize and train the logistic regression classifier
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

# Predict the target variable for the test set
Y_pred = classifier.predict(X_test)

# Calculate the confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate the accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Predict the target variable for a new sample
Y_pred = classifier.predict(X_test)

# Get predicted and actual labels
predicted_labels = le.inverse_transform(Y_pred)
actual_labels = le.inverse_transform(Y_test)

# Plotting the first two features
plt.scatter(X_test[:, 0], X_test[:, 1], c=Y_pred, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Predictions')
plt.show()

print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), 1))

# Example prediction for a new sample
print(classifier.predict(sc.transform([[5, 3.6, 1.4, 0.2]])))
