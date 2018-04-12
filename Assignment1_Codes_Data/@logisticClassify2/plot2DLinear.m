function plot2DLinear(obj, X, Y)
% plot2DLinear(obj, X,Y)
%   plot a linear classifier (data and decision boundary) when features X are 2-dim
%   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
%
  [n,d] = size(X);
  if (d~=2); error('Sorry -- plot2DLogistic only works on 2D data...'); end;

  %% Plot X seperately (by Y)
  classes = getClasses(obj); % get the classes
  figure;
  hold on;
  
  for i = 1:size(classes) % loop over the classes
      plot(X(Y==classes(i),1), X(Y==classes(i),2), '.'); % plot each point in this class
  end;
  
  %% Plot decision boundary
  %b = sign(getWeights(obj).*[ones(size(X,1),1), X]);
  wts = getWeights(obj);
  x = linspace(min(X(:,1)), max(X(:,1))); 
  y = -(wts(2)/wts(3))*x - (wts(1)/wts(3)); % re-arranged formula of theta'X=0
  
  p = plot(x,y); % plotting in a way that doesn't change axis
  set(p, 'YLimInclude', 'off');
  
  %% Some Beautification
  xlabel('Sepal Length');
  ylabel('Sepal Width');
  title('Decision Boundry');
  legend('Class 1', 'Class 2', 'Decision Boundry');