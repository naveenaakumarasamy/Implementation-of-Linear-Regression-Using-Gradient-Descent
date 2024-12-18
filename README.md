# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load Data: Read the dataset using pd.read_csv("50_Startups.csv")
2. Prepare Features and Target: Separate features x and target y from the dataset.
3. Scale Data: Use StandardScaler() to standardize x and y for better convergence.
4. Add Intercept: Add a column of ones to x for the intercept term.
5. Initialize Parameters: Set initial weights 0 = 0.
6. Prediction: Use the formula ŷ = X 0 to compute predictions.
7. Error: Calculate error as the difference between predicted values and actual values.
8. Compute Gradient: Find the gradient using 1 XT (X0 – у). - m
9. Update Parameters: Update weights using 0 = 0 - a gradient.
10. Repeat: Loop for a specified number of iterations (num_iters).
11. Make Predictions on New Data: Scale new data and compute predictions using updated weights.
12. Inverse Transform: Convert predictions be to the original scale

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: Naveenaa A K
RegisterNumber: 212222223094 

```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("50_Startups.csv")
print(data.head())
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```


## Output:
![image](https://github.com/user-attachments/assets/94f9ef7c-19fa-4271-a6d4-1c4ad8653270)
![image](https://github.com/user-attachments/assets/3854a2bf-7119-428a-957f-161d2425b85a)
![image](https://github.com/user-attachments/assets/f7486697-663b-4e15-92d8-bedb077628f1)
![image](https://github.com/user-attachments/assets/5d4baab6-3e3d-4b04-8bd0-4f4d65659798)
![image](https://github.com/user-attachments/assets/e83ded40-1f34-401a-af28-3303030eeb59)
![image](https://github.com/user-attachments/assets/54d3ed2e-49bd-4b30-a396-ff0c765ad3db)
![image](https://github.com/user-attachments/assets/ea802458-d285-4135-a165-63a36df27818)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
