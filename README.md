# ML_LinReg_GradDesc_Chess
Machine Learning using Linear Regression with Gradient Descent on Chess (King-Rook vs. King) Data Set
There is two parts which are separated to two different python files. The first one implements linear regression with gradient descent on the chess dataset where the 
algorithms are developed from scratch. The second part uses the scikit-learn library to implement linear regression with gradient descent.
The dataset being used is hosted from my github webpage and is from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Chess+%28King-Rook+vs.+King%29
The dataset is the data points for how many moves it takes for the white to win using optimal moves based on the positions of white king, white rook and black king. 
The models are setup for linear regression in an x vs y style taking x as the positions and y as the depth of win.

The attributes in the dataset are as follows:
1. White King file (column)
2. White King rank (row)
3. White Rook file
4. White Rook rank
5. Black King file
6. Black King rank
7. optimal depth-of-win for White in 0 to 16 moves, otherwise drawn {draw, zero, one, two, ..., sixteen}.

The dependencies for the first part include:
1. numpy
2. pandas
3. sklearn.metrics import r^2
4. sklearn.preprocessing import labelencoder
5. sklearn.model_selection import train_test_split
6. sklearn.metrics import rmse

The dependencies for the second part include:
1. numpy
2. pandas
3. sklearn.metrics import r^2
4. sklearn.preprocessing import labelencoder
5. sklearn.model_selection import train_test_split
6. sklearn.metrics import rmse
7. sklearn.linear_model import LinearRegression
8. sklearn.linear_model import SGDRegressor
