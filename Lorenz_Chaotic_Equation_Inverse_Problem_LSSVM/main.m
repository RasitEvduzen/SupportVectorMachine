% Lorenz System Parameter Estimation via Batch LS-SVM
% dx1/dt = a*(x2-x1)
% dx2/dt = b*x1 - x2 - x1*x3
% dx3/dt = x1*x2 - c*x3
% Written By: Rasit Evduzen
% Date: 27-Apr-2026

clc; clear; close all;

% --- Setup ---
t0 = 0; tf = 50; dt = 0.04;
t  = (t0:dt:tf).';
x0 = [-2; -1; 1];
theta_true = [10; 28; 8/3];

fprintf('True:  a=%.4f  b=%.4f  c=%.4f\n\n', theta_true(1), theta_true(2), theta_true(3));

% --- Simulate Lorenz ---
opts = odeset('RelTol',1e-5,'AbsTol',1e-5*ones(1,3));
sol  = ode45(@lorenz_ode, [t0 tf], x0, opts, theta_true);
Y    = deval(sol, t).';

noise_level = 0.01;
rng(1);
y1 = Y(:,1) + noise_level*randn(size(t));
y2 = Y(:,2) + noise_level*randn(size(t));
y3 = Y(:,3) + noise_level*randn(size(t));

% --- Hyperparameter grid search (K-fold CV) ---
K_fold       = 3;
gamma_range  = logspace(0, 6, 10);
sigma_range  = logspace(-3, 1, 10);
num_real     = 3;

Par = zeros(num_real, 3);

for itr = 1:num_real
    fprintf('Iteration %d / %d\n', itr, num_real);
    n   = numel(t);
    ind = crossvalind('Kfold', n, K_fold);

    BB1 = zeros(10,10); BB2 = BB1; BB3 = BB1;
    for gi = 1:10
        for si = 1:10
            gam = gamma_range(gi); sig = sigma_range(si);
            e1 = zeros(K_fold,1); e2 = e1; e3 = e1;
            for f = 1:K_fold
                tr = ind~=f; te = ind==f;
                [a1,b1,a2,b2,a3,b3] = lssvm_fit(t(tr), y1(tr), y2(tr), y3(tr), gam, sig);
                K2 = rbf_kernel(t(tr), sig, t(te));
                e1(f) = mean((y1(te) - K2'*a1 - b1).^2);
                e2(f) = mean((y2(te) - K2'*a2 - b2).^2);
                e3(f) = mean((y3(te) - K2'*a3 - b3).^2);
            end
            BB1(gi,si)=mean(e1); BB2(gi,si)=mean(e2); BB3(gi,si)=mean(e3);
        end
    end

    % --- Best params and full fit for each state ---
    [a1,b1,sig1,tnew,Kyt1] = best_fit(BB1, gamma_range, sigma_range, t, y1, t0, tf, dt);
    [a2,b2,sig2,   ~,Kyt2] = best_fit(BB2, gamma_range, sigma_range, t, y2, t0, tf, dt);
    [a3,b3,sig3,   ~,Kyt3] = best_fit(BB3, gamma_range, sigma_range, t, y3, t0, tf, dt);

    yhat1 = rbf_kernel(t, sig1, tnew)' * a1 + b1;
    yhat2 = rbf_kernel(t, sig2, tnew)' * a2 + b2;
    yhat3 = rbf_kernel(t, sig3, tnew)' * a3 + b3;
    ydot1 = Kyt1' * a1;
    ydot2 = Kyt2' * a2;
    ydot3 = Kyt3' * a3;

    % --- Linear system for [a, b, c] ---
    A = [(yhat2-yhat1), zeros(size(tnew)), zeros(size(tnew)); ...
          zeros(size(tnew)), yhat1,         zeros(size(tnew)); ...
          zeros(size(tnew)), zeros(size(tnew)), -yhat3];
    B = [ydot1; ydot2 + yhat2 + yhat1.*yhat3; ydot3 - yhat1.*yhat2];
    theta_hat = A \ B;

    fprintf('  est: a=%.4f  b=%.4f  c=%.4f\n', theta_hat(1), theta_hat(2), theta_hat(3));
    Par(itr,:) = theta_hat.';
end

theta_est = mean(Par).';
fprintf('\nFinal: a=%.4f  b=%.4f  c=%.4f\n', theta_est(1), theta_est(2), theta_est(3));
fprintf('Error: a=%.2e  b=%.2e  c=%.2e\n', abs(theta_est-theta_true));

% --- Simulate with estimated parameters ---
sol_est  = ode45(@lorenz_ode, [t0 tf], x0, opts, theta_est);
sol_true = ode45(@lorenz_ode, [t0 tf], x0, opts, theta_true);
Yest  = deval(sol_est,  tnew).';
Ytrue = deval(sol_true, tnew).';

% --- Plot ---
figure('units','normalized','outerposition',[0 0 1 1],'color','w');
labels = {'x_1(t)','x_2(t)','x_3(t)'};
for i = 1:3
    subplot(3,2, 2*i-1);
    plot(tnew, Ytrue(:,i), 'k', 'LineWidth', 2); hold on; grid minor;
    plot(tnew, Yest(:,i),  'r--', 'LineWidth', 2);
    ylabel(labels{i}); xlabel('t');
    legend('True','Estimated');
end
subplot(3,2,[2 4 6]);
plot3(Ytrue(:,1), Ytrue(:,2), Ytrue(:,3), 'k', 'LineWidth', 2); hold on;
plot3(Yest(:,1),  Yest(:,2),  Yest(:,3),  'r--', 'LineWidth', 1.5);
xlabel('x_1'); ylabel('x_2'); zlabel('x_3');
title('Phase Space'); grid on;
legend('True','Estimated');


%% ----------------------- Local functions -----------------------

function dy = lorenz_ode(~, y, p)
    dy = [p(1)*(y(2)-y(1)); p(2)*y(1)-y(2)-y(1)*y(3); y(1)*y(2)-p(3)*y(3)];
end

function K = rbf_kernel(Xtr, sig, Xt)
    if nargin < 3, Xt = Xtr; end
    D2 = bsxfun(@plus, sum(Xtr.^2,2), sum(Xt.^2,2).') - 2*(Xtr*Xt.');
    K  = exp(-D2 / sig);
end

function [a1,b1,a2,b2,a3,b3] = lssvm_fit(Xtr, y1, y2, y3, gam, sig)
    K = rbf_kernel(Xtr, sig);
    m = size(K,1);
    A = [K + (1/gam)*eye(m), ones(m,1); ones(1,m), 0];
    r1=A\[y1;0]; a1=r1(1:m); b1=r1(end);
    r2=A\[y2;0]; a2=r2(1:m); b2=r2(end);
    r3=A\[y3;0]; a3=r3(1:m); b3=r3(end);
end

function [alpha, bias, sig, tnew, Kyt] = best_fit(BB, gam_range, sig_range, t, y, t0, tf, dt)
    [~, idx] = min(BB(:));
    [p, q]   = ind2sub(size(BB), idx);
    gam = gam_range(p); sig = sig_range(q);
    K   = rbf_kernel(t, sig);
    m   = size(K,1);
    A   = [K + (1/gam)*eye(m), ones(m,1); ones(1,m), 0];
    res = A \ [y; 0];
    alpha = res(1:m); bias = res(end);
    tnew  = (t0 : dt/2 : tf).';
    Knew  = rbf_kernel(t, sig, tnew);
    % Analytical derivative of RBF: dK/dt = -2*(t-tnew')/sig * K
    Kyt   = (2/sig) .* bsxfun(@minus, t, tnew.') .* Knew;
end