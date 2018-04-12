%% 5. Perceptions and Logistic Regression
%% Importing data
% Clean up
clear;
clc;
close all;
% Import 
iris = load('data/iris.txt');
X = iris(:,1:2); Y = iris(:,end); % use the first two features and classifier
% Reformat data
[X, Y] = shuffleData(X,Y); % shuffle the data 
X = rescale(X);
XA = X(Y<2,:); YA=Y(Y<2); % split into classes 0 and 1
XB = X(Y>0,:); YB=Y(Y>0); % and 1 and 2

%% Scatter plot of seperable and non-seperable data
% Plot class 0 vs 1
figure; 
plot(XA(:,1), XA(:,2), 'b.');
title('Class 0 vs Class 1 (Separable)');
xlabel('Sepal Length');
ylabel('Sepal Width');
% Plot class 1 vs 2
figure;
plot(XB(:,1), XB(:,2), 'b.');
title('Class 1 vs Class 2 (Non-Separable)');
xlabel('Sepal Length');
ylabel('Sepal Width');

%% Plot decision boundry using provided weights

wts = [0.5 1 -0.25]; % set up the weights
learnerA = logisticClassify2(); % create "blank" learners
learnerB = logisticClassify2();
learnerA = setClasses(learnerA, unique(YA)); % define class labels 
learnerB = setClasses(learnerB, unique(YB));

learnerA=setWeights(learnerA, wts); % set the learner's parameters
learnerB=setWeights(learnerB, wts);
plot2DLinear(learnerA, XA, YA); % plot
plot2DLinear(learnerB, XB, YB); 


%% Calculate error rate
errorRateA = err(learnerA, XA, YA)
errorRateB = err(learnerB, XB, YB)
