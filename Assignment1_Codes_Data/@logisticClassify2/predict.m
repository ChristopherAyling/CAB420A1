function Yte = predict(obj,Xte)
% Yhat = predict(obj, X)  : make predictions on test data X

% (1) make predictions based on the sign of wts(1) + wts(2)*x(:,1) + ...
% (2) convert predictions to saved classes: Yte = obj.classes( [1 or 2] );
[n,d] = size(Xte);

wts = getWeights(obj);
X1 = [ones(n,1), Xte];
Yte = sign(wts*X1')';
Yte(Yte==-1) = obj.classes(1);
Yte(Yte==1) = obj.classes(2);