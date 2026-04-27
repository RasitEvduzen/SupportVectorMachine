% AOSVR - Accurate Online Support Vector Regression
% Ma, Theiler & Perkins (2003) - nonlinear regression demo on noisy sinc data.
% Written By: Rasit Evduzen
% Date: 27-Apr-2026
clc; clear; close all;
%%
% --- Data ---
N = 1000;
rng(1);
X      = linspace(-10, 10, N).';
Y_true = sinc(X / pi);                  % clean true function
Y      = Y_true + 0.02*randn(N, 1);      % noisy training data

% Random presentation order for online training
perm = randperm(N);
Xord = X(perm, :);
Yord = Y(perm);

% --- Hyperparameters ---
kp.type  = 'rbf';
kp.gamma = 1;
C        = 20;
eps_     = 1e-4;

% --- Train and plot ---
model = aosvr_init(Xord(1:2,:), Yord(1:2), C, eps_, kp);

figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w');
for k = 3:N
    model = aosvr_add(model, Xord(k,:), Yord(k));

    if mod(k, 50) == 0
        Yhat = aosvr_predict(model, X);

        clf; hold on;
        scatter(X, Y, 20, [0.5 0.5 0.5], 'filled');        % noisy training data
        plot(X, Y_true, 'r', 'LineWidth', 3);            % true function
        plot(X, Yhat,   'b', 'LineWidth', 2);              % AOSVR fit
        title(sprintf('AOSVR Nonlinear Regression  (n=%d)', k));
        xlabel('X'); ylabel('Y');
        legend({'Noisy samples', 'True f(x)', 'AOSVR f(x)'}, 'Location', 'northeast');
        grid on;axis([-10 10 -0.4 1.2])
        drawnow;
        
    end
end


%% ----------------------- Local functions -----------------------

function K = kernel(X1, X2, kp)
    if isempty(X1) || isempty(X2)
        K = zeros(size(X1,1), size(X2,1));
        return;
    end
    switch lower(kp.type)
        case 'rbf'
            sq1 = sum(X1.^2, 2);
            sq2 = sum(X2.^2, 2);
            D2  = bsxfun(@plus, sq1, sq2.') - 2*(X1*X2.');
            K   = exp(-kp.gamma * max(D2, 0));
        case 'linear'
            K = X1 * X2.';
    end
end


function f = aosvr_predict(model, X)
    active = [model.setS(:); model.setE(:)];
    if isempty(active)
        f = model.b * ones(size(X,1),1);
        return;
    end
    f = kernel(X, model.Xtrain(active,:), model.kp) * model.theta(active) + model.b;
end


function model = aosvr_init(X, y, C, eps_, kp)
    if y(1) >= y(2)
        x1 = X(1,:); y1 = y(1); x2 = X(2,:); y2 = y(2); swp = false;
    else
        x1 = X(2,:); y1 = y(2); x2 = X(1,:); y2 = y(1); swp = true;
    end

    K11 = kernel(x1, x1, kp);
    K12 = kernel(x1, x2, kp);
    denom = 2*(K11 - K12);
    if abs(denom) < 1e-12
        theta1 = 0;
    else
        theta1 = max(0, min(C, (y1 - y2 - 2*eps_) / denom));
    end
    if swp
        theta = [-theta1; theta1];
    else
        theta = [ theta1; -theta1];
    end

    model.Xtrain = X(1:2,:);
    model.ytrain = y(1:2);
    model.theta  = theta;
    model.b      = (y1 + y2)/2;
    model.C      = C;
    model.eps    = eps_;
    model.kp     = kp;
    model.N      = 2;

    TOL = 1e-9;
    model.setS = []; model.setE = []; model.setR = [];
    for i = 1:2
        ti = model.theta(i);
        if abs(ti) <= TOL
            model.setR = [model.setR; i];
        elseif abs(abs(ti) - C) <= TOL
            model.setE = [model.setE; i];
        else
            model.setS = [model.setS; i];
        end
    end

    model.h = aosvr_predict(model, model.Xtrain) - model.ytrain;

    if isempty(model.setS)
        model.Rmat = [];
    else
        S = model.setS;
        Q_SS = kernel(model.Xtrain(S,:), model.Xtrain(S,:), kp);
        model.Rmat = inv([0 ones(1,numel(S)); ones(numel(S),1) Q_SS]);
    end
end


function model = aosvr_add(model, xc, yc)
    C = model.C; eps_ = model.eps;
    TOL = 1e-9; MAX_ITERS = 10000;

    c = model.N + 1;
    model.Xtrain(c,:) = xc;
    model.ytrain(c,1) = yc;
    model.theta(c,1)  = 0;
    model.h(c,1)      = 0;
    model.N           = c;

    hc = aosvr_predict(model, xc) - yc;
    model.h(c) = hc;

    if abs(hc) <= eps_ + TOL
        model.setR = [model.setR; c];
        return;
    end

    q = -sign(hc);

    for iter = 1:MAX_ITERS
        S = model.setS; E = model.setE; Rset = model.setR;
        nS = numel(S);

        if nS > 0
            Q_Sc    = kernel(model.Xtrain(S,:), xc, model.kp);
            beta    = -model.Rmat * [1; Q_Sc];
            Q_cc    = kernel(xc, xc, model.kp);
            gamma_c = Q_cc + Q_Sc.' * beta(2:end) + beta(1);
        else
            beta    = 1;
            gamma_c = 1;
        end

        cand = zeros(0,3);

        target_hc = -q*eps_;
        if abs(gamma_c) > TOL
            L = (target_hc - model.h(c)) / gamma_c;
            if q*L > 0, cand(end+1,:) = [abs(L), 1, c]; end
        end

        L = q*C - model.theta(c);
        if q*L > 0, cand(end+1,:) = [abs(L), 2, c]; end

        for k = 1:nS
            i = S(k); bsi = beta(k+1); ti = model.theta(i);
            if abs(bsi) < TOL, continue; end
            if q*bsi > 0
                if ti < 0, L = (0 - ti)/bsi; if q*L>0, cand(end+1,:)=[abs(L),3,i]; end, end
                if ti < C, L = (C - ti)/bsi; if q*L>0, cand(end+1,:)=[abs(L),3,i]; end, end
            else
                if ti >  0, L = ( 0-ti)/bsi; if q*L>0, cand(end+1,:)=[abs(L),3,i]; end, end
                if ti > -C, L = (-C-ti)/bsi; if q*L>0, cand(end+1,:)=[abs(L),3,i]; end, end
            end
        end

        ER = [E(:); Rset(:)];
        gER = zeros(numel(ER),1);
        if ~isempty(ER)
            Q_ic_all = kernel(model.Xtrain(ER,:), xc, model.kp);
            if nS > 0
                Q_iS_all = kernel(model.Xtrain(ER,:), model.Xtrain(S,:), model.kp);
                gER = Q_ic_all + Q_iS_all*beta(2:end) + beta(1);
            else
                gER = ones(numel(ER),1);
            end
        end

        for k = 1:numel(E)
            i = E(k); gi = gER(k);
            if abs(gi) < TOL, continue; end
            target_h = sign(model.theta(i))*eps_;
            L = (target_h - model.h(i))/gi;
            if q*L > 0, cand(end+1,:) = [abs(L), 4, i]; end
        end

        for k = 1:numel(Rset)
            i = Rset(k); gi = gER(numel(E) + k);
            if abs(gi) < TOL, continue; end
            for target_h = [eps_, -eps_]
                L = (target_h - model.h(i))/gi;
                if q*L > 0, cand(end+1,:) = [abs(L), 5, i]; end
            end
        end

        if isempty(cand)
            warning('aosvr_add: no candidate at iter %d', iter);
            return;
        end

        [Lmin, idx] = min(cand(:,1));
        flag = cand(idx,2);
        I    = cand(idx,3);
        dth  = q * Lmin;

        if nS > 0
            model.b        = model.b + beta(1)*dth;
            model.theta(S) = model.theta(S) + beta(2:end)*dth;
        else
            model.b        = model.b + dth;
        end
        model.theta(c) = model.theta(c) + dth;

        model.h(c) = model.h(c) + gamma_c*dth;
        if ~isempty(ER)
            model.h(ER) = model.h(ER) + gER*dth;
        end

        switch flag
            case 1
                model.h(c) = target_hc;
                model.Rmat = R_add(model.Rmat, beta, gamma_c, kernel(xc, xc, model.kp));
                model.setS = [model.setS; c];
                return;

            case 2
                model.theta(c) = q*C;
                model.setE = [model.setE; c];
                return;

            case 3
                tI   = model.theta(I);
                kpos = find(S == I, 1);
                model.Rmat = R_remove(model.Rmat, kpos+1);
                model.setS(model.setS == I) = [];
                if abs(tI) > C - TOL
                    model.theta(I) = sign(tI)*C;
                    model.setE = [model.setE; I];
                else
                    model.theta(I) = 0;
                    model.setR = [model.setR; I];
                end

            case 4
                model.h(I) = sign(model.theta(I))*eps_;
                [bI, gI]   = beta_gamma_for(model, I);
                K_II       = kernel(model.Xtrain(I,:), model.Xtrain(I,:), model.kp);
                model.Rmat = R_add(model.Rmat, bI, gI, K_II);
                model.setE(model.setE == I) = [];
                model.setS = [model.setS; I];

            case 5
                if abs(model.h(I) - eps_) < abs(model.h(I) + eps_)
                    model.h(I) =  eps_;
                else
                    model.h(I) = -eps_;
                end
                [bI, gI]   = beta_gamma_for(model, I);
                K_II       = kernel(model.Xtrain(I,:), model.Xtrain(I,:), model.kp);
                model.Rmat = R_add(model.Rmat, bI, gI, K_II);
                model.setR(model.setR == I) = [];
                model.setS = [model.setS; I];
        end
    end
    warning('aosvr_add: hit MAX_ITERS');
end


function [bI, gI] = beta_gamma_for(model, I)
    S = model.setS;
    if isempty(S)
        bI = 1; gI = 1; return;
    end
    Q_IS = kernel(model.Xtrain(I,:), model.Xtrain(S,:), model.kp);
    bI   = -model.Rmat * [1; Q_IS.'];
    Q_II = kernel(model.Xtrain(I,:), model.Xtrain(I,:), model.kp);
    gI   = Q_II + Q_IS*bI(2:end) + bI(1);
end


function Rnew = R_add(R, beta, gamma_, K_II)
    if isempty(R)
        Rnew = [-K_II 1; 1 0];
        return;
    end
    if abs(gamma_) < 1e-12, gamma_ = 1e-12; end
    n   = size(R,1);
    ext = [R, zeros(n,1); zeros(1, n+1)];
    v   = [beta(:); 1];
    Rnew = ext + (1/gamma_) * (v*v.');
end


function Rnew = R_remove(R, k)
    if size(R,1) == 2
        Rnew = [];
        return;
    end
    p = R(k,k);
    if abs(p) < 1e-12, p = 1e-12; end
    Rnew = R - (R(:,k) * R(k,:)) / p;
    Rnew(k,:) = [];
    Rnew(:,k) = [];
end