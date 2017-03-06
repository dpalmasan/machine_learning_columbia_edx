# Project 1: Linear Regression

This project consisted on implementing Ridge Regression algorithm and an active learning scheme.

* hw1_regression.py: Contains two functions, one for computing weights of Ridge Regression, and one for extracting 10 indexes of potentially interesting data to measure output (based on variance).

* testing_project.py: Used for testing purposes, to ensure the solution obtained is correct.

* hw1_regression.m: The solution, but in a source code in octave.

* RidgeRegression.m: An octave function for ridge regression, returns the weights given training data and the regularization hyperparameter lambda.ç

* ActiveLearning.m: Applies an active learning scheme for selecting 10 potentially interesting data, from test set. Returns de indexes corresponding to samples found.



##To execute the code:

`python hw1_regression.py lambda sigma2 <X_train.csv> <y_train.csv> <X_test.csv>`

`octave hw1_regression.m -q lambda sigma2 <X_train.csv> <y_train.csv> <X_test.csv>`




