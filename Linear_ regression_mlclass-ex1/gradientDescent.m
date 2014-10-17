function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

T1 =( X * theta) - y;
T1 = sum(T1); 
T1 = (T1 * alpha)/m;
theta_temp(1)= theta(1) - T1;


T2 =( X * theta) - y;
T2 = T2' * X(:,2); 
T2 = (T2 * alpha)/m;
theta_temp(2) = theta(2) - T2;
%fprintf('Avant %f %f \n', theta(1), theta(2));

theta(1) = theta_temp(1);
theta(2) =  theta_temp(2);
%fprintf('Apres %f %f \n', theta(1), theta(2));

J = computeCost(X, y, theta);
%fprintf('Cost %f \n', J);

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
    
end

end
