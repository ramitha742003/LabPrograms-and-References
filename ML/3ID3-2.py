from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Create Decision Tree classifier with 'entropy' (ID3-like)
clf = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42)
clf.fit(X_train, y_train)

# 4. Predict on test set
y_pred = clf.predict(X_test)

# 5. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Predictions:", y_pred)
print("Actual     :", y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# 6. Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=feature_names, class_names=target_names)
plt.title("Decision Tree using sklearn (ID3-like)")
plt.show()
