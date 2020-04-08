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


% ----- PART 1 -----

%X = 5000*400
a1 = [ones(size(X,1), 1) X]; %5000*401

z2 = a1 * Theta1'; %z2 = 5000*25
a2 = sigmoid(z2);
a2 = [ones(size(a2, 1), 1) a2]; %5000*26 

z3 = a2 * Theta2'; % z3 = 5000x10
a3 = sigmoid(z3);

%convert y from vector to matrix
y_matrix = eye(num_labels)(y,:);  %5000x10

J = (1/m)*sum(sum((-y_matrix.*log(a3)-(1-y_matrix).*log(1-a3))));

% Regularized cost function
rTheta1 = Theta1(:, 2:end);
rTheta2 = Theta2(:, 2:end);

J = J + (lambda/(2*m))*(sum(sum((rTheta1.*rTheta1))) + sum(sum((rTheta2.*rTheta2))));

% ----- PART 2 -----

Theta2_d2 = Theta2(:, 2:end);
Theta1_grad = 0;
Theta2_grad = 0;

for t = 1:m
  %step 1
  a_1 = X(t,:); % 1x400
  a_1 = a_1'; %400x1
  a_1 = [1 ; a_1]; %add bias unit, 401x1
  z_2 = Theta1 * a_1; %25x1
  a_2 = sigmoid(z_2); %25x1
  a_2 = [1 ; a_2]; %26x1
  z_3 = Theta2 * a_2; %10x1
  a_3 = sigmoid(z_3); %10x1
  %Step 2
  y_k = y_matrix(t,:); %1x10
  y_k = y_k'; %10x1
  d_3 = a_3 - y_k; %10x1
  fprintf("");
  %Step 3
  d_2 = (Theta2_d2') * d_3 .* sigmoidGradient(z_2); %(10x25)' * 10x1 .* 25x1 = 25x1
  %Step 4
  Delta1 = d_2 * (a_1)'; % (25x1) * (401x1)' = 25x401
  Delta2 = d_3 * (a_2)'; % (10x1) * (26x1)' = 10x26
  Theta1_grad = Theta1_grad + Delta1;
  Theta2_grad = Theta2_grad + Delta2;
endfor

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
