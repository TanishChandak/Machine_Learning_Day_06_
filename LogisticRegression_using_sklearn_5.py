import pandas as pd
import matplotlib.pyplot as plt

# loading the data:
df = pd.read_csv('insurance_data.csv')
print(df)

# ploting the graph:
plt.scatter(x=df.age, y=df.bought_insurance, marker='+', color='red')
plt.show()

# training and testing the dataset:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['age']], df.bought_insurance, test_size=0.1)
print("It will give the data which is for the test: ",X_test)
print("It will give the data which is for the train: ",X_train)

# Logistic Regression:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# training the model:
model.fit(X_train, y_train)
print(model.predict(X_test))
print("The accuracy of the model: ",model.score(X_test, y_test))
print(model.predict_proba(X_test))