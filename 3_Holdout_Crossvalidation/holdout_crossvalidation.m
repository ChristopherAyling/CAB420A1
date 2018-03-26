%% 4. Hold-out and Cross-validation
%% Import data
%Clean up
clc
clear
close all
%Import
mTrain = load('data/mcycleTrain.txt');
ytr = mTrain(:,1); 
xtr = mTrain(:,2);