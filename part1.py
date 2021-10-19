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

from sklearn.metrics import mean_squared_error

#calculate coefficient of variation using numpy and pandas
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100

#gradient descent
def gradient_descent(grad, start, learn=0.1, iterate=15, toler=1e-08):
    vect = start
    for x in range(iterate):
        differ = -learn * np.array(grad(vect))
        if np.all(np.abs(differ) <= toler):
            break
        vect += differ
        print(vect)
    return vect

b0, b1, b2, b3, b4, b5, b6 = df.apply(cv)
lin_model = np.array(b0 + b1 * np.asarray(Y_train))
predict = gradient_descent(grad=lambda Y_train: np.array(b0 + b1 * np.asarray(Y_train)), start=[0.5, 0.5], learn=0.0001, iterate=15)

# model evaluation for training set
y_train_predict = predict[0] + np.dot(predict[1], Y_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

predict = gradient_descent(grad=lambda Y_test: np.array(b0 + b1 * np.asarray(Y_test)), start=[0.5, 0.5], learn=0.0001, iterate=15)
# model evaluation for testing set

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

y_test_predict =  predict[0] + np.dot(predict[1], Y_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))