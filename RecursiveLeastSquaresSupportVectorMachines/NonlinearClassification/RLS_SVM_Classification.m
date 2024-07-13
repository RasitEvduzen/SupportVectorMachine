% Spiral data generation and RLS-SVM based classification
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

%% RLS-SVM
% Random start RLSE state vector
x_rlse = rand(size(A, 1), 1);
P = 1e2*eye(size(A, 1), size(A, 1));

figure('units', 'normalized', 'outerposition', [0 0 .3 .5], 'color', 'w')
for k = 1:N
    [x_rlse, K, P] = rlse_online(A(k, :), b(k, :), x_rlse, P);
    bias = x_rlse(1);
    alpha = x_rlse(2:end);   % Support Vectors!
    if mod(k,20) == 0
        [X1, X2] = meshgrid(-1:0.02:1, -1:0.02:1);
        classifier = @(x) sign(sum(alpha .* Y .* arrayfun(@(i) kernel(X(i,:), x,gamma), 1:N)') + bias);
        Z = arrayfun(@(x1, x2) classifier([x1, x2]), X1, X2);
        clf
        hold on
%         for i = 1:numel(X1)
%             if Z(i) == 1
%                 plot(X1(i), X2(i), 'y.', 'MarkerSize', 10);
%             else
%                 plot(X1(i), X2(i), 'b.', 'MarkerSize', 10);
%             end
%         end
        contourf(X1, X2, Z, 1,'--'),hold on;
        scatter(X(Y==1,1), X(Y==1,2), 'k', "filled");
        scatter(X(Y==-1,1), X(Y==-1,2), 'r', "filled");
        title('RLS-SVM Nonlinear Classification');
        xlabel('X1');
        ylabel('X2');
        drawnow
    end
end


function [x, K, P] = rlse_online(a_k, b_k, x, P)
% One step of RLSE (Recursive Least Squares Estimation) algorithm
a_k = a_k(:);
b_k = b_k(:);
K = (P * a_k) / (a_k' * P * a_k + 1); % Compute Gain K (Like Kalman Gain!)
x = x + K * (b_k - a_k' * x);         % State Update
P = P - K * a_k' * P;                 % Covariance Update
end