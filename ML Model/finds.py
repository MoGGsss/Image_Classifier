from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Generate a random dataset
X, y = make_classification(n_samples=1000)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Create a multi-layer perceptron classifier with one hidden layer of 5 neurons
mlp = MLPClassifier( max_iter=1000, solver='sgd', random_state=42)
# Train the model using backpropagation
mlp.fit(X_train, y_train)
# Calculate the accuracy score of the model
accuracy = mlp.score(X_test, y_test)
print(f"Accuracy: {accuracy}")