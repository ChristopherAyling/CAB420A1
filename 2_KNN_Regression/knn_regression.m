%% 2. KNN Regression
%% Load Data
%Clean up
clc
clear
close all
%Load train data
mTrain = load('mcycleTrain.txt');
Ytrain = mTrain(:,1);
Xtrain = mTrain(:,2);
%Load test data
mTest = load('mcycleTest.txt');
Ytest = mTest(:,1);
Xtest = mTest(:,2);
%Plot data
figure('name', 'Motorcycle Data');
hold on
plot(Xtrain, Ytrain, 'bo');
plot(Xtest, Ytest, 'ro');
legend('Training data', 'Test data');
hold off

%% a) Implement predict function
learner = knnRegress(1,Xtrain, Ytrain);
Yhat = predict(learner, Xtest);
figure('name', 'Knn Testing');
hold on
plot(Xtest, Ytest, 'ro');
plot(Xtest, Yhat, 'bo');
hold off

%% b) Plot function for several values of K


%% c) What kind of functions can be output?