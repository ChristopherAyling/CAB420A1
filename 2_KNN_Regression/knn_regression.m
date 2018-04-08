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
legend('Training data');
hold off

%% a) Implement predict function
learner = knnRegress(1,Xtrain, Ytrain);
Yhat = predict(learner, Xtest);
figure('name', 'Knn Testing');
hold on
plot(Xtest, Ytest, 'go');
plot(Xtest, Yhat, 'ro');
legend('Y test', 'Y hat');
hold off

%% b) Plot function for several values of K
ks = [1, 2, 3, 4, 5, 10, 50];
Xs = min(Xtrain):0.001:max(Xtrain); Xs = Xs';
figure('name', 'testing values of k')
hold on
learner = knnRegress(1,Xtrain, Ytrain);
for i=1:length(ks)
    learner = knnRegress(ks(i),Xtrain, Ytrain);
    stairs(Xs, predict(learner, Xs), '-', 'linewidth', 5);
end
title('kNN for values of K')
legend(cellstr(int2str(ks')))
hold off

%% c) What kind of functions can be output?
