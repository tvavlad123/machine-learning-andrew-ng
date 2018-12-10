function g = sigmoid(z)

%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.
% You need to return the following variables correctly 

g = zeros(size(z));
f=@(h) 1 ./ (1 + exp(-h)); %anonymous function
g = f(z);

end
