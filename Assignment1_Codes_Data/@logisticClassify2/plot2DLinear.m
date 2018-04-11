function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(X);
  if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;

  %% Plot X seperately (by Y)
  figure;
  plot(X(Y==1,2), X(Y==1,1), 'b.');
  hold on;
  plot(X(Y~=1,2), X(Y~=1,1), 'r.');
  
  %% Plot decision boundary
  prediction = predict(obj, X);
  