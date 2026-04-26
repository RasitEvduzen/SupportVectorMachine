% AOSVR - Accurate Online Support Vector Regression
% Ma, Theiler & Perkins (2003), adapted for binary classification:
% Written By: Rasit Evduzen
% Date: 26-Apr-2026
clc; clear; close all;
%%
% --- Spiral data ---
B = 8; N = 500;
Tall = [];
for i = 1:N/2
    th   = pi/2 + (i-1)*((2*B-1)/N)*pi;
    Tall = [Tall, [th*cos(th); th*sin(th)]];
end
Tall = [Tall, -Tall];
Tmax = pi/2 + ((N/2-1)*(2*B-1)/N)*pi;
X    = Tall.' / Tmax;
Y    = [-ones(N/2,1); ones(N/2,1)];

rng(1);
perm = randperm(N);
X = X(perm,:); Y = Y(perm);

% --- Hyperparameters ---
kp.type  = 'rbf';
kp.gamma = 100;
C        = 100;
eps_     = 0.1;

% --- Training order: start with one + and one - sample ---
ipos = find(Y ==  1, 1);
ineg = find(Y == -1, 1);
order = [ipos; ineg; setdiff((1:N).', [ipos;ineg], 'stable')];
Xord = X(order,:);
Yord = Y(order);

% --- Train and plot ---
model = aosvr_init(Xord(1:2,:), Yord(1:2), C, eps_, kp);

figure('units', 'normalized', 'outerposition', [0 0 1 1], 'color', 'w');
for k = 3:N
    model = aosvr_add(model, Xord(k,:), Yord(k));

    if mod(k, 10) == 0
        [X1, X2] = meshgrid(-1:0.02:1, -1:0.02:1);
        Z = aosvr_predict(model, [X1(:), X2(:)]);
        Z = reshape(sign(Z), size(X1));

        clf; hold on;
        contourf(X1, X2, Z, 1);
        scatter(Xord(Yord== 1, 1), Xord(Yord== 1, 2), 'k', 'filled');
        scatter(Xord(Yord==-1, 1), Xord(Yord==-1, 2), 'r', 'filled');
        title('AOSVR Nonlinear Classification');
        xlabel('X1'); ylabel('X2');
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
    % Two-sample analytic init (paper Eq. 26). Requires y(1) >= y(2);
    % we swap if needed and map theta back.
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

    % Initial set membership
    TOL = 1e-9;
    model.setS = []; model.setE = []; model.setR = [];
    for i = 1:2
        ti = model.theta(i);
        if abs(ti) <= TOL
            model.setR(end+1,1) = i;
        elseif abs(abs(ti) - C) <= TOL
            model.setE(end+1,1) = i;
        else
            model.setS(end+1,1) = i;
        end
    end

    model.h = aosvr_predict(model, model.Xtrain) - model.ytrain;

    % R matrix = inv([0 1...; 1 Q_SS])
    if isempty(model.setS)
        model.Rmat = [];
    else
        S = model.setS;
        Q_SS = kernel(model.Xtrain(S,:), model.Xtrain(S,:), kp);
        model.Rmat = inv([0 ones(1,numel(S)); ones(numel(S),1) Q_SS]);
    end
end


function model = aosvr_add(model, xc, yc)
    % Incremental update: add (xc, yc) so the model matches the batch SVR
    % solution on the augmented training set.
    C = model.C; eps_ = model.eps;
    TOL = 1e-9; MAX_ITERS = 10000;

    % Append new sample with theta_c = 0
    c = model.N + 1;
    model.Xtrain(c,:) = xc;
    model.ytrain(c,1) = yc;
    model.theta(c,1)  = 0;
    model.h(c,1)      = 0;
    model.N           = c;

    hc = aosvr_predict(model, xc) - yc;
    model.h(c) = hc;

    % Already inside the eps-tube: assign to R, done
    if abs(hc) <= eps_ + TOL
        model.setR(end+1,1) = c;
        return;
    end

    q = -sign(hc);   % direction of dtheta_c

    for iter = 1:MAX_ITERS
        S = model.setS; E = model.setE; Rset = model.setR;
        nS = numel(S);

        % --- beta and gamma_c (paper Eqs. 19, 20b) ---
        if nS > 0
            Q_Sc    = kernel(model.Xtrain(S,:), xc, model.kp);
            beta    = -model.Rmat * [1; Q_Sc];
            Q_cc    = kernel(xc, xc, model.kp);
            gamma_c = Q_cc + Q_Sc.' * beta(2:end) + beta(1);
        else
            beta    = 1;
            gamma_c = 1;
        end

        % --- Bookkeeping: collect all candidate |L| values ---
        cand = zeros(0,3);   % columns: |L|, flag, sample index

        % Case 1: xc reaches margin -> joins S
        target_hc = -q*eps_;
        if abs(gamma_c) > TOL
            L = (target_hc - model.h(c)) / gamma_c;
            if q*L > 0, cand(end+1,:) = [abs(L), 1, c]; end
        end

        % Case 2: theta_c reaches q*C -> joins E
        L = q*C - model.theta(c);
        if q*L > 0, cand(end+1,:) = [abs(L), 2, c]; end

        % Case 3: a sample in S leaves S (toward 0 or +/-C)
        for k = 1:nS
            i = S(k); bsi = beta(k+1); ti = model.theta(i);
            if abs(bsi) < TOL, continue; end
            if q*bsi > 0      % theta_i increasing
                if ti < 0, L = (0 - ti)/bsi; if q*L>0, cand(end+1,:)=[abs(L),3,i]; end, end
                if ti < C, L = (C - ti)/bsi; if q*L>0, cand(end+1,:)=[abs(L),3,i]; end, end
            else              % theta_i decreasing
                if ti >  0, L = ( 0-ti)/bsi; if q*L>0, cand(end+1,:)=[abs(L),3,i]; end, end
                if ti > -C, L = (-C-ti)/bsi; if q*L>0, cand(end+1,:)=[abs(L),3,i]; end, end
            end
        end

        % gamma for each E and R sample
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

        % Case 4: i in E -> S (h_i reaches sign(theta_i)*eps)
        for k = 1:numel(E)
            i = E(k); gi = gER(k);
            if abs(gi) < TOL, continue; end
            target_h = sign(model.theta(i))*eps_;
            L = (target_h - model.h(i))/gi;
            if q*L > 0, cand(end+1,:) = [abs(L), 4, i]; end
        end

        % Case 5: i in R -> S (h_i reaches +eps or -eps)
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

        % --- Apply parameter updates ---
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

        % --- Set transitions and R-matrix update ---
        switch flag
            case 1   % xc joins S
                model.h(c) = target_hc;
                model.Rmat = R_add(model.Rmat, beta, gamma_c, kernel(xc, xc, model.kp));
                model.setS(end+1,1) = c;
                return;

            case 2   % xc joins E
                model.theta(c) = q*C;
                model.setE(end+1,1) = c;
                return;

            case 3   % I in S leaves S
                tI   = model.theta(I);
                kpos = find(S == I, 1);
                model.Rmat = R_remove(model.Rmat, kpos+1);
                model.setS(model.setS == I) = [];
                if abs(tI) > C - TOL
                    model.theta(I) = sign(tI)*C;
                    model.setE(end+1,1) = I;
                else
                    model.theta(I) = 0;
                    model.setR(end+1,1) = I;
                end

            case 4   % I in E joins S
                model.h(I) = sign(model.theta(I))*eps_;
                [bI, gI]   = beta_gamma_for(model, I);
                K_II       = kernel(model.Xtrain(I,:), model.Xtrain(I,:), model.kp);
                model.Rmat = R_add(model.Rmat, bI, gI, K_II);
                model.setE(model.setE == I) = [];
                model.setS(end+1,1) = I;

            case 5   % I in R joins S
                if abs(model.h(I) - eps_) < abs(model.h(I) + eps_)
                    model.h(I) =  eps_;
                else
                    model.h(I) = -eps_;
                end
                [bI, gI]   = beta_gamma_for(model, I);
                K_II       = kernel(model.Xtrain(I,:), model.Xtrain(I,:), model.kp);
                model.Rmat = R_add(model.Rmat, bI, gI, K_II);
                model.setR(model.setR == I) = [];
                model.setS(end+1,1) = I;
        end
    end
    warning('aosvr_add: hit MAX_ITERS');
end


function [bI, gI] = beta_gamma_for(model, I)
    % beta and gamma for sample I about to enter S, using the current S
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
    % Eq. 25: extend R when a sample joins S.
    if isempty(R)
        Rnew = [-K_II 1; 1 0];   % inv([0 1; 1 K_II])
        return;
    end
    if abs(gamma_) < 1e-12, gamma_ = 1e-12; end
    n   = size(R,1);
    ext = [R, zeros(n,1); zeros(1, n+1)];
    v   = [beta(:); 1];
    Rnew = ext + (1/gamma_) * (v*v.');
end


function Rnew = R_remove(R, k)
    % Eq. 24: shrink R when sample at position k leaves S (k is row/col in R).
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