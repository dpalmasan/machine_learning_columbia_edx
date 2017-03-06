arg_list = argv();

lambda = str2double(arg_list{1});
sigma2 = str2double(arg_list{2});
X_train = csvread(arg_list{3});
y_train = csvread(arg_list{4});
X_test = csvread(arg_list{5});

wRR = RidgeRegression(X_train, y_train, lambda);
indexes = ActiveLearning(X_train, X_test, lambda, sigma2);
format long
disp(wRR)
disp(indexes)
