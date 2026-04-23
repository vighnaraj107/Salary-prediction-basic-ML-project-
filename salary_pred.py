import pandas as pd
from matplotlib import pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#loading the data
data=pd.read_csv('Salary_Data.csv')
#print(data.head())

#naming the x and y variables
x=data[['YearsExperience']]
y=data['Salary']

#splitting the data into training and testing data
X_train,X_test,y_train,y_test=train_test_split(x,
y,test_size=0.2,random_state=42)

#training the model
model=LinearRegression()
model.fit(X_train,y_train)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

#predicting the values
predictions=model.predict(X_test)

#printing the predictions
print("Predictions:", predictions)

#calculating the R-squared score
r2=r2_score(y_test,predictions)
print("R-squared Score:", r2)

#detailed comparison of actual and predicted values
print("\nDetailed Comparison:")
for i in range(len(X_test)):
    print("Experience:", X_test.iloc[i, 0],
          "| Actual:", y_test.iloc[i],
          "| Predicted:", predictions[i])

#plot graph to visualize the data
plt.scatter(x,y,color='blue')
plt.plot(X_test,predictions,color='red')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Experience')
plt.show()