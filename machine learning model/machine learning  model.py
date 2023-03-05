# Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("/content/titaniccleaned_data (7).csv")

# Extract features and target variable
X = df[['Name', 'Sex', 'Fare', 'Cabin', 'Embarked']]
y = df['Survived']

# Clean data
X = X.fillna('Unknown')

# Convert categorical variables to numerical
X = pd.get_dummies(X)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a decision tree model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Predict target variable using decision tree model
y_pred = decision_tree.predict(X_test)

# Evaluate accuracy of decision tree model
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Accuracy: {accuracy}")

# Create a random forest model
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

# Predict target variable using random forest model
y_pred = random_forest.predict(X_test)

# Evaluate accuracy of random forest model
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy}")