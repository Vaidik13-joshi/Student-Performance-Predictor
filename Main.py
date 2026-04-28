import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create dataset
data = {
    'Study_Hours': [1,2,3,4,5,6,7,8],
    'Attendance': [50,55,60,65,70,75,80,85],
    'Result': [0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Study_Hours', 'Attendance']]
Y = df['Result']

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Predictions
Y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:\n", cm)

# New student prediction
new_student = pd.DataFrame([[6, 72]], columns=['Study_Hours', 'Attendance'])
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction: Pass")
else:
    print("Prediction: Fail")