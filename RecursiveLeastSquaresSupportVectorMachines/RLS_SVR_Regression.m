% Non linear function using Recursive Least Squares Support Vector Regression method
% Written By: Rasit
% 07-Mar-2024
clc,clear all,close all;
%% Input and Output definition
xtrain = [1:0.1:20]';
NoD=length(xtrain);

% ytrain = sin(xtrain) + 2*sin(2*xtrain)+0.5*randn(NoD,1);
ytrain = 0.01*xtrain.*xtrain + 0.1*exp(-xtrain) + sin(xtrain) +0.1*randn(NoD,1);

%% Train RLS-SVM
K = zeros(NoD,NoD);
C = 1000;     % Over Fitting Param (C=100)
gamma = 5e-1; % RBF param, equal to 1/2sigma^2 (g=1e-2)

kernelSelect = 'rbf';
K = Kernel(kernelSelect,xtrain,gamma);

A=[0, ones(1,NoD);
    ones(NoD,1), (K + 1/C*eye(NoD))];
b = [0; ytrain];
%% RLS-SVR
% xlse = inv(A'*A)*A'*b;   % Least Squares Solution for Support Vector Regression! Offline Solution!

x_rlse = rand(size(A,1),1);  % Random start RLSE state vector
P = 1e2 * eye(size(A,1),size(A,1));

% Prediction Stage
xpred = xtrain; % Prediction Input
ypred = zeros(NoD,1);
tmp = zeros(NoD,1);

figure('units','normalized','outerposition',[0 0 1 1],'color','w')
for k=1:NoD
    [x_rlse,K,P] = rlse_online(A(k,:),b(k,:),x_rlse,P);
    % Bias and alpha LaGrange Multipliers
    bias = x_rlse(1);
    alpha = x_rlse(2:end);

    for j=1:NoD
        for i=1:NoD
            tmp(i,1) = alpha(i,1).*exp(-gamma*(xpred(j,1)-xtrain(i,1))^2);
        end
        ypred(j,1) = sum(tmp) + bias;
    end
    % Plot Result
    clf
    plot(xtrain, ytrain,'ko-',LineWidth=2),hold on;
    plot(xpred, ypred,'r.-',LineWidth=2);
    axis([0 21 -2 5.5])
    title('RLS-SVR Regression');
    legend('Noisy Data', 'RLS-SVR Output');
    drawnow
end


function [x,K,P] = rlse_online(a_k,b_k,x,P)
% One step of RLSE (Recursive Least Squares Estimation) algorithm
a_k = a_k(:);
b_k = b_k(:);
K = (P*a_k)/(a_k'*P*a_k+1); % Compute Gain K (Like Kalman Gain!)
x = x + K*(b_k-a_k'*x);     % State Update
P = P - K*a_k'*P;           % Covariance Update
end
