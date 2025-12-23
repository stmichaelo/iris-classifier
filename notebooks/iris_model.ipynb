#load iris data
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
print(iris.feature_names, iris.target_names)

#Split data into training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#80/20 split for train/test

#initialise the model using decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

#train the model
model.fit(X_train, y_train)

#predict the species of the flower
y_pred = model.predict(X_test)
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])

#evaluate the performance of the predictions
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#check the decision tree performance using the KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
y_true = [1,0,2,1,1]
y_pred = [1,0,2,1,1]
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot().figure_.savefig('confusion_matrix.png')

import joblib
joblib.dump(model, "model.joblib")
