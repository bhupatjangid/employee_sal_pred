import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('slr2.csv')
real_x=data.iloc[:,[0]].values
real_y=data.iloc[:,[1]].values


from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.3,random_state=0)

from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(train_x,train_y)
pred_y=lin.predict(test_x)
#print(test_y)
#print(pred_y)

plt.scatter(train_x,train_y,color='green')
#plt.show()
plt.scatter(train_x,train_y,color='green')
plt.plot(train_x,lin.predict(train_x))
plt.show()
plt.scatter(test_x,test_y,color='blue')
plt.plot(train_x,lin.predict(train_x))
plt.show()