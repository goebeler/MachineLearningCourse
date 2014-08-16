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
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%---------------------------Feedforward and Unregularized Cost Function------------------------

X = [ones(m, 1) X];		% Add bias units to the input layer
y = eye(num_labels)(y,:); 	% Change labels to vectors containig 0 and 1
a1 = X;				% Input layer is actually the first layer
z2 = a1 * Theta1';		% Calculate hypothesis for the first hidden layer
a2 = sigmoid(z2);		% Apply sigmoid function on the first hidden layer
n = size(a2, 1);		% Get count of rows in the first hidden layer
a2 = [ones(n, 1) a2];		% Add bias units to the first hidden layer
z3 = a2 * Theta2';		% Calculate hypothesis for the second hidden layer
a3 = sigmoid(z3);		% Apply sigmoid function on the second hidden layer


% Calculate cost function without regularization
innerSumExpression = (-y .* log(a3)) - ((1 - y) .* log(1 - a3));
J = (1/m) * sum(sum(innerSumExpression));	

%----------------------------------------------------------------------------------------------
%---------------------------Regularized Cost Function------------------------------------------

Theta1_NoBiasUnits = Theta1(:,2:end);		% Removing bias units from Theta1
Theta2_NoBiasUnits = Theta2(:,2:end);		% Removing bias units from Theta2

% Calculate regularization for the cost function
regularization = (lambda / (2 * m)) * (sum(sum(Theta1_NoBiasUnits .^ 2)) + sum(sum(Theta2_NoBiasUnits .^ 2)));

% Calculate cost function with regularization
J = J + regularization;

%----------------------------------------------------------------------------------------------
%---------------------------Neural Net Gradient Function (Backpropagation)---------------------

% Compute "error" of nodes in second and in first hidden layer
delta_3 = a3 - y;
delta_2 = (delta_3 * Theta2_NoBiasUnits) .* sigmoidGradient(z2);

% Compute gradient for the nodes in the second and first hidden layers
delta_cap2 = delta_3' * a2;
delta_cap1 = delta_2' * a1;

% Obtain the (unregularized) gradient for the neural network cost function
Theta1_grad = ((1/m) * delta_cap1);
Theta2_grad = ((1/m) * delta_cap2);
 
%---------------------------------------------------------------------------
% --------------------------Regularized Gradient-----------------------------------------------

% Obtain the (regularized) gradient for the neural network cost function
Theta1_grad = ((1/m) * delta_cap1) + (lambda/m) * Theta1;
Theta2_grad = ((1/m) * delta_cap2) + (lambda/m) * Theta2;

% NOTE: We cannot remove the bias units because we will have 25x401 and 25x400 dimensional 
% matrices which we cannot sum together. That is why we do it now by subtracting them from 
% the gradient matrices,because we should not regularize them.  
Theta1_grad(:,1) -= ((lambda/m) * (Theta1(:,1)));
Theta2_grad(:,1) -= ((lambda/m) * (Theta2(:,1)));

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
