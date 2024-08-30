import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from joblib import load
import matplotlib.pyplot as plt

# Load the trained SVM model from the saved file
svm_model = load('svm_model_weights.joblib')

# Generate a new synthetic dataset for prediction (unrelated to the original training data)
X_new, y_new = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Standardize the new data (use the same scaling as the model was trained on)
scaler = StandardScaler()
X_new = scaler.fit_transform(X_new)

# Use the loaded model to make predictions on the new data
y_pred_new = svm_model.predict(X_new)

# Evaluate the model on the new data
print("Confusion Matrix:")
cm = confusion_matrix(y_new, y_pred_new)
print(cm)

print("\nClassification Report:")
report = classification_report(y_new, y_pred_new)
print(report)

accuracy = accuracy_score(y_new, y_pred_new)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the new data points with their predicted labels
plt.figure(figsize=(8, 6))
plt.scatter(X_new[:, 0], X_new[:, 1], c=y_pred_new, cmap='viridis', edgecolor='k')
plt.title("Predicted Labels for New Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print("Predicted labels for the new data points:")
print(y_pred_new)
