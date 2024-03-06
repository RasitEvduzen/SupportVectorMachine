% Non linear function using Least Squares Support Vector Regression method
% Written By: Rasit
% 06-Mar-2024
clc,clear all,close all;
%% Input and Output definition
xtrain = [1:0.1:20]';
NoD=length(xtrain);

% ytrain = sin(xtrain) + 2*sin(2*xtrain)+0.5*randn(NoD,1);
ytrain = 0.01*xtrain.*xtrain + 0.1*exp(-xtrain) + sin(xtrain) +0.1*randn(NoD,1);

%% Train LS-SVM

K = zeros(NoD,NoD); 
C = 1000;     % Over Fitting Param (C=100)
gamma = 5e-1; % RBF param, equal to 1/2sigma^2 (g=1e-2)

kernelSelect = 'rbf';
K = Kernel(kernelSelect,xtrain,gamma);

A=[0, ones(1,NoD);
        ones(NoD,1), (K + 1/C*eye(NoD))];
b = [0; ytrain];

xlse = inv(A'*A)*A'*b;   % Least Squares Solution for Support Vector Regression!

% Bias and alpha LaGrange Multipliers 
bias = xlse(1);
alpha = xlse(2:end);

% Prediction Stage
xpred = xtrain; % Prediction Input

ypred = zeros(NoD,1);
tmp = zeros(NoD,1);

for j=1:NoD
 for i=1:NoD
     tmp(i,1) = alpha(i,1).*exp(-gamma*(xpred(j,1)-xtrain(i,1))^2);
 end
     ypred(j,1) = sum(tmp) + bias;
end

%% Plot Result
figure('units','normalized','outerposition',[0 0 1 1],'color','w')
plot(xtrain, ytrain,'ko-',LineWidth=2),hold on; 
plot(xpred, ypred,'r.-',LineWidth=2);
title('LS-SVR Regression');
legend('Noisy Samples', 'Approximated');

% Plot Kernel Result
% figure
% [X,Y] = meshgrid(xtrain,xtrain);
% surf(X,Y,K,EdgeColor="none")
