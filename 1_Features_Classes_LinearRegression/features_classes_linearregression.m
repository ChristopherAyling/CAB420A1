%% (a)
%Load in and set up variables
mTrain = load('data/mcycleTrain.txt');
ytr = mTrain(:,1); 
xtr = mTrain(:,2);
%Plot the training data
plot(xtr, ytr, 'bo');
hold on;
legend('Training data');

%% (b)
linXtr = polyx(xtr, 1);
linLearner = linearReg(linXtr, ytr);
xline = [0:.01:2]';
yline = predict(linLearner, polyx(xline, 1));
plot(xline, yline);
legend('Training data', 'Linear predictor');

%% (c)
figure;
plot(xtr, ytr, 'bo');
hold on;
fifthXtr = polyx(xtr, 5);
fifthLearner = linearReg(fifthXtr, ytr);
xline = [0:.01:2]';
yline = predict(fifthLearner, polyx(xline, 5));
plot(xline, yline);
legend('Training data', 'Fifth degree polynomial');
axis([0 2 -150 100]);

%% (d)
training_MSE_of_linear_predictor = mse(linLearner, linXtr, ytr)
training_MSE_of_fifth_degree_predictor = mse(fifthLearner, fifthXtr, ytr)


%% (e)
mTest = load('data/mcycleTest.txt');
ytst = mTest(:,1);
xtst = mTest(:,2);
linXtst = polyx(xtst, 1);
fifthXtst = polyx(xtst, 5);

test_MSE_of_linear_predictor = mse(linLearner, linXtst, ytst)
test_MSE_of_fifth_degree_predictor = mse(fifthLearner, fifthXtst, ytst)
