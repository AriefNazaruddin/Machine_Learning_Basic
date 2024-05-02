import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['figure.dpi'] = 100

df = pd.read_csv("data/homeprices.csv")
print(df)

plt.scatter(df.area, df.price)
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Home Prices')
plt.show()
