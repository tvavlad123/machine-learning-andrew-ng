function [J, gradient] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

m = length(y); % number of training examples


J = 0;
gradient = zeros(size(theta));
predictions =  sigmoid(X * theta);
first = -y' * log(predictions);
second = (1 - y') * log(1 - predictions);
J = (1 / m) * (first - second);
gradient = (1 / m) * ((predictions - y)' * X);
	
end
