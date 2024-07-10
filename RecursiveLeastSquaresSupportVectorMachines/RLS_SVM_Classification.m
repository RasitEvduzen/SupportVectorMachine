% Spiral data generation and RLS-SVM classification using Recursive Least Squares
% Written By: Rasit
% 07-Mar-2024
clc, clear all, close all;

%% Generate Spiral Data
B = 4;
N = 200;
Tall = [];
for i = 1:N/2
    theta = pi/2 + (i-1)*[(2*B-1)/N]*pi;
    Tall = [Tall , [theta*cos(theta); theta*sin(theta)]];
end
Tall = [Tall, -Tall];
Tmax = pi/2 + [(N/2-1)*(2*B-1)/N]*pi;
xtrain = [Tall]'/Tmax;
ytrain = [-ones(1, N/2), ones(1, N/2)]';
NoD = length(xtrain);

%% Train RLS-SVM
K = zeros(NoD, NoD);
C = 1000;     % Over Fitting Param (C=100)
gamma = 5e-2; % RBF param, equal to 1/2sigma^2 (g=1e-2)

kernelSelect = 'rbf';
K = Kernel(kernelSelect, xtrain, gamma);

A = [0, ones(1, NoD);
     ones(NoD, 1), (K + 1/C * eye(NoD))];
b = [0; ytrain];

%% RLS-SVM
% Random start RLSE state vector
x_rlse = rand(size(A, 1), 1);  
P = 1 * eye(size(A, 1), size(A, 1));

% Training Stage
for k = 1:NoD
    [x_rlse, K, P] = rlse_online(A(k, :), b(k, :), x_rlse, P);
end

% Bias and alpha LaGrange Multipliers
bias = x_rlse(1);
alpha = x_rlse(2:end);

%% Test Data Generation
[x1, x2] = meshgrid(linspace(-1, 1, 100), linspace(-1, 1, 100));
xtest = [x1(:), x2(:)];
NoD_test = size(xtest, 1);

%% Prediction Stage
ypred_test = zeros(NoD_test, 1);
for j = 1:NoD_test
    tmp = zeros(NoD, 1);
    for i = 1:NoD
        tmp(i, 1) = alpha(i, 1) .* exp(-gamma * sum((xtest(j, :) - xtrain(i, :)).^2));
    end
    ypred_test(j, 1) = sign(sum(tmp) + bias);
end

%% Plot Result
figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w')
gscatter(xtrain(:,1), xtrain(:,2), ytrain, 'yk');
hold on;
gscatter(xtest(:,1), xtest(:,2), ypred_test, 'rb', '.');
title('RLS-SVM Classification on Spiral Data');
legend('Class 1', 'Class 2', 'Class 1 Prediction', 'Class 2 Prediction');
axis equal
drawnow

function [x, K, P] = rlse_online(a_k, b_k, x, P)
% One step of RLSE (Recursive Least Squares Estimation) algorithm
a_k = a_k(:);
b_k = b_k(:);
K = (P * a_k) / (a_k' * P * a_k + 1); % Compute Gain K (Like Kalman Gain!)
x = x + K * (b_k - a_k' * x);         % State Update
P = P - K * a_k' * P;                 % Covariance Update
end
