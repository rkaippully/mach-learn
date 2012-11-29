%
%   Copyright (c) 2012, Raghu Kaippully
%   All rights reserved.
%   
%   Redistribution and use in source and binary forms, with or without
%   modification, are permitted provided that the following conditions are met:
%       * Redistributions of source code must retain the above copyright
%         notice, this list of conditions and the following disclaimer.
%       * Redistributions in binary form must reproduce the above copyright
%         notice, this list of conditions and the following disclaimer in the
%         documentation and/or other materials provided with the distribution.
%       * Neither the name of the <organization> nor the
%         names of its contributors may be used to endorse or promote products
%         derived from this software without specific prior written permission.
%   
%   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
%   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
%   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%   DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
%   DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
%   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
%   LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
%   ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
%   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
%   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%

%{
    Linear regression - Gradient Descent

    http://mach-learn.blogspot.in/

    This program performs a linear regression using gradient descent.
%}

%{
    This is the feature matrix X representing the number of months since June 2008.
    We have data for 30 months, so X contains numbers 1 to 30. X is a 30x1 matrix.
    Each row in the matrix represents one training sample.
%}
X = (1:30)';

%{
    Y contains the lines of code in millions in Mozilla Firefox project from June 2008
    to November 2010 (30 months). Y is a 30x1 matrix.
%}
Y = [
4.028506;
4.065591;
4.093264;
4.070544;
4.095775;
4.130883;
4.247049;
4.296544;
4.306611;
4.312506;
4.288533;
4.370491;
4.379101;
4.397162;
4.390944;
4.559444;
4.645135;
4.718766;
4.734918;
4.771165;
4.812666;
4.865136;
4.910578;
4.910224;
5.065474;
5.155591;
5.249963;
5.267284;
5.378627;
5.342854;
];

%{
    Plot the data in a graph
%}
fprintf('Plotting the training data set...\n');
plot(X, Y, 'rx');
xlabel('No. of months');
ylabel('Lines of code (in millions)');
hold on;

%{
    Add x0 as the first column to X. x0 = 1 for all training samples.
%}
m = size(X, 1);         % m is the number of training examples - size of X along the first dimension
X = [ones(m, 1), X];    % add a column of ones to X
n = size(X, 2);         % n is the number of features - size of X along the second dimension

%{
    Initialize theta0 and theta1 to zeros
%}
theta = zeros(n, 1);

%{
    And these are our gradient descent settings
%}
iterations = 8000;
alpha = 0.005;

%{
    Run gradient descent
%}
fprintf('Running gradient descent...\n');
cost = zeros(iterations, 1);
for iter = 1:iterations
    fprintf('\rStep %d of %d', iter, iterations);

    % First compute the value of h(theta) at the current value of theta
    hypotheses = X*theta;
    
    %{
        Find the step size by computing the gradient at this point. This computes
        (h(X) - Y) * X. Here step is an mxn matrix.
    %}
    step = bsxfun(@times, X, (hypotheses - Y));

    % Take a step: theta = theta - alpha * (1/m) * step
    theta = theta - alpha * mean(step)';

    % Save the cost function at each step
    cost(iter) = (X*theta - Y)'*(X*theta - Y)/(2*m);
end

fprintf('\ntheta(0) = %f, theta(1) = %f.\n', theta(1), theta(2));

%{
    Plot the linear fit using theta
%}
plot(X(:, 2), X*theta, 'b');

%{
    Plot the cost as a function of iterations
%}
figure(2);
plot(cost);
xlabel('Iterations');
ylabel('Cost function');

fprintf('Done.\n');
