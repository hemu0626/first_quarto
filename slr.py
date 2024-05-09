import matplotlib.pyplot as plt
import numpy as np

X = np.array([0.1, 0.2, 0.3, 
              0.4, 0.5, 0.6, 
              0.7, 0.8, 0.9, 1]).reshape(-1,1)

y = np.array([0.05,0.08,0.1,
             0.09,0.13,0.14,
             0.17,0.21,0.28,0.27]).reshape(-1,1)
plt.scatter(X,y)
