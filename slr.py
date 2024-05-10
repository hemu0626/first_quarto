import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
X = np.array([0.1, 0.2, 0.3, 
              0.4, 0.5, 0.6, 
              0.7, 0.8, 0.9, 1]).reshape(-1,1)

y = np.array([0.05,0.08,0.1,
             0.09,0.13,0.14,
             0.17,0.21,0.28,0.27]).reshape(-1,1)
plt.scatter(X,y)
plt.show()


slr = LinearRegression()
slr.fit(X, y)
predict_y = slr.predict(X)
print('slope:',slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

y_predict = slr.predict(X)
# plot results
plt.scatter(X, y, label='training points')
plt.plot(X, y_predict, label='linear fit', linestyle='--')
plt.show()

mse = np.mean((y-y_predict)**2)
print("MSE=",mse)

from sklearn.metrics import mean_squared_error
print('MSE train: %.6f' %mean_squared_error(y, predict_y))
