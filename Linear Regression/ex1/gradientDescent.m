function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    aux1 = theta(1) - ( alpha / m ) * sum(((X * theta) - y).* X(:, 1));
    aux2 = theta(2) - ( alpha / m ) * sum(((X * theta) - y).* X(:, 2));
    theta = [aux1; aux2];
    J_history(iter) = computeCost(X, y, theta);

end

end
