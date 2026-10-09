#Decision Tree Classification

# Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv("Iris.csv")

# Separate features and target
X = data.drop(columns=["Id", "Species"], errors="ignore")
y = data["Species"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the decision tree classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict the species
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision
precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# Calculate recall
recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

# Display evaluation results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Display classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))