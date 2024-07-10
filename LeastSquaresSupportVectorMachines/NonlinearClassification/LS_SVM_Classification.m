% Spiral data generation and LS-SVM based classification 
% Written By: Rasit
% 07-Mar-2024
clc, clear all, close all;
%% Data Generation
B = 4;
N = 200;
Tall = [];
for i = 1:N/2
    theta = pi/2 + (i-1)*[(2*B-1)/N]*pi;
    Tall = [Tall , [theta*cos(theta); theta*sin(theta)]];
end
Tall = [Tall, -Tall];
Tmax = pi/2 + [(N/2-1)*(2*B-1)/N]*pi;

X = [Tall]'/Tmax;
Y = [-ones(1, N/2), ones(1, N/2)]';


%% Kernel Function
kernel = @(x, y,gamma) exp(-gamma*norm(x-y)^2);

% LS-SVM Param
gamma = 1e2; 
C = 1e2;

% Kernel Matrix
N = size(X, 1);
K = zeros(N, N);
for i = 1:N
    for j = 1:N
        K(i,j) = kernel(X(i,:), X(j,:),gamma);
    end
end


Y_diag = diag(Y);
Omega = Y_diag * K * Y_diag;
I = eye(N);

A = [0, Y'; Y, Omega + 1/C * I];
b = [0; ones(N, 1)];
x_lse = A\b;

bias = x_lse(1);
alpha = x_lse(2:end);   % Support Vectors!

%% Prediction

classifier = @(x) sign(sum(alpha .* Y .* arrayfun(@(i) kernel(X(i,:), x,gamma), 1:N)') + bias);
[X1, X2] = meshgrid(-1:0.1:1, -1:0.1:1);
Z = arrayfun(@(x1, x2) classifier([x1, x2]), X1, X2);

%% Plot
figure('units', 'normalized', 'outerposition', [0 0 .3 .5], 'color', 'w')
contourf(X1, X2, Z, 1,'--'),hold on;
scatter(X(Y==1,1), X(Y==1,2), 'k', "filled");
scatter(X(Y==-1,1), X(Y==-1,2), 'r', "filled");
title('LS-SVM Nonlinear Classification');
xlabel('X1');
ylabel('X2');
