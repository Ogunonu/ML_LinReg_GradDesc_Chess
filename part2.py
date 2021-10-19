import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

#parameters in order:
# 1. White King file (column)
# 2. White King rank (row)
# 3. White Rook file
# 4. White Rook rank
# 5. Black King file
# 6. Black King rank
# 7. optimal depth-of-win for White in 0 to 16 moves, otherwise drawn {draw, zero, one, two, ..., sixteen}.
#import data from my github hosted site
data = pd.read_csv("https://ogunonu.github.io/krkopt.data", sep=',', names=['WKF', 'WKR', 'WRF', 'WRR', 'BKF', 'BKR', 'Depth'])

#setup dataframe for pandas
df = pd.DataFrame(data, columns=('WKF', 'WKR', 'WRF', 'WRR', 'BKF', 'BKR', 'Depth'))

#preprocess categorical variables into numerical variables
number = LabelEncoder()
df['WKF'] = number.fit_transform(data['WKF'].astype('str'))
df['WRF'] = number.fit_transform(data['WRF'].astype('str'))
df['BKF'] = number.fit_transform(data['BKF'].astype('str'))
df['Depth'] = number.fit_transform(data['Depth'].astype('str'))

#test and train data
from sklearn.model_selection import train_test_split
X = pd.DataFrame(np.c_[df['WKF'], df['WKR'], df['WRF'], df['WRR'], df['BKF'], df['BKR']], columns = ['WKF', 'WKR', 'WRF', 'WRR', 'BKF', 'BKR'])
Y = df['Depth']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state = 5)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

from sklearn.linear_model import SGDRegressor

lin_model = SGDRegressor()
lin_model.fit(X_train, Y_train)
lin_model.coef_
lin_model.n_iter_no_change

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))