%% 2. KNN Regression
%% Load Data
mTrain = load('mcycleTrain.txt');
Ytrain = mTrain(:,1);
Xtrain = mTrain(:,2);

mTest = load('mcycleTest.txt');
Ytest = mTest(:,1);
Xtest = mTest(:,2);

figure('name', 'Motorcycle Data');
hold on
plot(Xtrain, Ytrain, 'bo');
plot(Xtest, Ytest, 'ro');
hold off

%% a) Implement predict function
learner = knnRegress(1,Xtrain, Ytrain);
Yhat = predict(learner, Xtest);
figure('name', 'Knn Testing');
hold on
plot(Xtest, Ytest, 'bo');
plot(Xtest, Yhat, 'ro');
hold off

%% b) Plot function for several values of K


%% c) What kind of functions can be output?