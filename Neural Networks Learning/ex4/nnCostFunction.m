function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


a1 = [ones(m, 1) X];
a2 = sigmoid(a1 * Theta1');
a2 = [ones(m, 1) a2];	
a3 = sigmoid(a2 * Theta2');
y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
J = (-1 / m) * sum(sum(y.* log(a3) + (1 - y).* log(1 - a3)));

regTheta1 =  Theta1(:,2:end);
regTheta2 =  Theta2(:,2:end);

error = (lambda / (2 * m)) * (sum(sum(regTheta1.^2)) + sum(sum(regTheta2.^2)));

J = J + error;
del1 = zeros(size(Theta1));
del2 = zeros(size(Theta2));
for t = 1:m,
	a1t = a1(t,:);
	a2t = a2(t,:);
	a3t = a3(t,:);
	yt = y(t,:);
	d3 = a3t - yt;
	d2 = Theta2'*d3' .* sigmoidGradient([1;Theta1 * a1t']);
	del1 = del1 + d2(2:end)*a1t;
	del2 = del2 + d3' * a2t;
end;

Theta1_grad = 1 / m * del1 + (lambda / m) * [zeros(size(Theta1, 1), 1) regTheta1];
Theta2_grad = 1 / m * del2 + (lambda / m) * [zeros(size(Theta2, 1), 1) regTheta2];
% -------------------------------------------------------------

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end