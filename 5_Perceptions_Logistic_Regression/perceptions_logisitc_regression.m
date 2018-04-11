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
plot(XA(:,2), XA(:,1), 'b.');
title('Class 0 vs Class 1 (Separable)');
xlabel('Sepal Length');
ylabel('Sepal Width');
% Plot class 1 vs 2
figure;
plot(XB(:,2), XB(:,1), 'b.');
title('Class 1 vs Class 2 (Non-Separable)');
xlabel('Sepal Length');
ylabel('Sepal Width');