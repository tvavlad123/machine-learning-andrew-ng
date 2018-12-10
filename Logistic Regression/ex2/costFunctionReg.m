function [J, gradient] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

J = 0;
gradient = zeros(size(theta));
predictions =  sigmoid(X * theta);
first = -y' * log(predictions);
second = (1 - y') * log(1 - predictions);
t0 = theta;
t0(1) = 0;
cost = (lambda / (2 * m)) * sum(t0 .^ 2);
aux = lambda / m * t0;
J = (1 / m) * (first - second) + cost;
gradient = ((1/m) * (X' * (predictions - y))) + aux;

end