#Data Collection and Exploration

#Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#Load the dataset into a pandas DataFrame
data = pd.read_csv('wine_quality_dataset.csv')

#Explore the dataset by examining its structure and statistical summary
print(data.head())
print(data.describe())

#Data Preprocessing:-

#Split the data into features (X) and target variable (y)
X = data.drop('quality', axis=1)
y = data['quality']

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Feature Scaling:-

#Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Model Training and Evaluation:-

#Train a Random Forest Classifier model
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

#Make predictions on the test set
y_pred = model.predict(X_test_scaled)

#Evaluate the model's accuracy:
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Prediction:-

#Prepare new, unseen wine data for prediction:
new_data = pd.DataFrame({
    'fixed acidity': [7.4],
    'volatile acidity': [0.7],
    'citric acid': [0.0],
    'residual sugar': [1.9],
    'chlorides': [0.076],
    'free sulfur dioxide': [11.0],
    'total sulfur dioxide': [34.0],
    'density': [0.9978],
    'pH': [3.51],
    'sulphates': [0.56],
    'alcohol': [9.4]
})

#Scale the new data using the same scaler:
new_data_scaled = scaler.transform(new_data)

#Predict the quality of the new data:
new_prediction = model.predict(new_data_scaled)
print("Predicted quality:", new_prediction)
