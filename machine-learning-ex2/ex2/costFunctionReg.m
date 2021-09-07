function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
temp0 = sigmoid(X*theta);
temp1 = log(temp0);
temp2 = log(1-temp0);
theta_new = theta(2:end);
J = (-y'*temp1-(1-y)'*temp2)./m + lambda/(2*m)*theta_new'*theta_new;
grad(1) = ((temp0-y)'*X(:,1))/m;
grad(2) = ((temp0-y)'*X(:,2))/m + lambda/m*theta(2) ;
grad(3) = ((temp0-y)'*X(:,3))/m + lambda/m*theta(3) ;




% =============================================================

end
