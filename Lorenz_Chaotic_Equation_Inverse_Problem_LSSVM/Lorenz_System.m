% dot(x1)  = a * (x_2 -x_1)
% dot(x_2) = x_1 * (b- x_3) - x_2
% dot(x_3) = x_1 * x_2 -c* x_3
% 0 <=  t  < = t_f
% Initial Condition
% x_1(0) = -9.42, x_2(0)= -9.34, x_3(0)=28.3
% Theta=[a, b, c] = [10, 28, 8/3]

clear all; close all; clc

t0 = 0;
tf = 50;
sampling_time = 0.04;
t = (t0:sampling_time:tf)';
% initial = [-9.42 -9.34 28.3]; % initial values of the ODE used for generating simulated data
initial = [-2 -1 1]; % initial values of the ODE used for generating simulated data
% initial = [-5 -7.1 15.3]; % initial values of the ODE used for generating simulated data
ExactTheta = [10; 28 ; 8/3];  % The exact parameters of the lorenz system used for generating simulated data

cprintf( [1 0.1 0],'**** Excat parameters of the Lorenz system ***** \n\n');
fprintf('True theta_1= %f \n', ExactTheta(1));
fprintf('True theta_2= %f \n', ExactTheta(2));
fprintf('True theta_3= %f \n\n', ExactTheta(3));
fprintf( '************************************* \n\n');

%%  ========= Generating the simulation data ======================

options = odeset('RelTol',1e-5,'AbsTol',[1e-5 1e-5 1e-5]);
sol = ode45(@ridg,[t0 tf],initial,options,ExactTheta);
Y = deval(sol,t);
Y = Y';

noise_level = 0.01; % 0.03, 0.05, 0.07, 0.1
noise = noise_level*randn(size(t,1),1);

figure
subplot(3,2,[2,4,6])
plot3(Y(:,1),Y(:,2),Y(:,3),'r')
grid minor
set(gcf,'Position',[10 250 750 500])
xlabel('x_1(t)','FontSize', 24)
ylabel('x_2(t)','FontSize', 24)
zlabel('x_3(t)','FontSize', 24)
title('Phase-Space','FontSize', 24 )

set(gcf,'Position',[250 250 1500 700])
subplot(3,2,1)
plot(t,Y(:,1),'b')
grid minor
xlabel('t','FontSize', 24)
ylabel('x_1(t)','FontSize', 24)

subplot(3,2,3)
plot(t,Y(:,2),'b')
grid minor
xlabel('t','FontSize', 24)
ylabel('x_2(t)','FontSize', 24)

subplot(3,2,5)
plot(t,Y(:,3),'b')
grid minor
xlabel('t','FontSize', 24)
ylabel('x_3(t)','FontSize', 24)
y1 = Y(:,1)+noise;
y2 = Y(:,2)+noise;
y3 = Y(:,3)+noise;

% y1 = Y(:,1);
% y2 = Y(:,2);
% y3 = Y(:,3);


%% Estimating the parameters of the ODE system:

num_realization  = 3;
K_fold = 3;
num_grid_gam = 10;
num_grid_sig = 10;
gamma_range = logspace(0,6,num_grid_gam);
sigma_range = logspace(-3,1,num_grid_sig);
ER1 = [];
ER2 = [];
ER3 = [];
Par1 = zeros(num_realization,1);
Par2 = zeros(num_realization,1);
Par3 = zeros(num_realization,1);
BB1 = zeros(num_grid_gam,num_grid_sig);
BB2 = zeros(num_grid_gam,num_grid_sig);
BB3 = zeros(num_grid_gam,num_grid_sig);
for itr=1:num_realization
    
    cprintf( [1 0.1 0],'**** iteration  = %d\n',itr);
    n = size(t,1);
    ind = crossvalind('Kfold', n, K_fold);
    
    for gamma_idx = 1:size(gamma_range,2)
        gamma  =  gamma_range(gamma_idx);
        
        for sig_idx = 1:size(sigma_range,2)
            sig = sigma_range(sig_idx);
            
            test_errors_1 = zeros(K_fold,1);
            test_errors_2 = zeros(K_fold,1);
            test_errors_3 = zeros(K_fold,1);
            
            
            for i = 1:K_fold
                Xte = t(ind ==i,1);
                Yte_1 = y1(ind==i,1);
                Yte_2 = y2(ind==i,1);
                Yte_3 = y3(ind==i,1);
                Xtr = t(ind~=i,1);
                Ytr_1 = y1(ind~=i,1);
                Ytr_2 = y2(ind~=i,1);
                Ytr_3 = y3(ind~=i,1);
                               
                K = KernelMatrix(Xtr,'RBF_kernel', sig);
                m = size(K,1);
                A =  [K + (1/gamma) * eye(m), ones(m,1);...
                    ones(m,1)' ,0];
                B1 =  [Ytr_1;0];
                B2 =  [Ytr_2;0];
                B3 =  [Ytr_3;0];
                result1 = A\B1;
                result2 = A\B2;
                result3 = A\B3;
                alpha1 = result1(1:m);
                b1 = result1(end);
                alpha2 = result2(1:m);
                b2 = result2(end);
                alpha3 = result3(1:m);
                b3 = result3(end);
                yhattr1 = K * alpha1 + b1;
                yhattr2 = K * alpha2 + b2;
                yhattr3 = K * alpha3 + b3;
                
                K2 = KernelMatrix(Xtr,'RBF_kernel', sig, Xte);
                yhatte1 = K2' * alpha1  + b1;
                yhatte2 = K2' * alpha2  + b2;
                yhatte3 = K2' * alpha3  + b3;
                etest1 = Yte_1-yhatte1 ;
                etest2 = Yte_2-yhatte2 ;
                etest3 = Yte_3-yhatte3 ;
                test_errors_1(i) = mse(etest1);
                test_errors_2(i) =  mse(etest2);
                test_errors_3(i) = mse(etest3);
                
            end
            
            mean_testerror_1 = mean(test_errors_1);
            mean_testerror_2 = mean(test_errors_2);
            mean_testerror_3 = mean(test_errors_3);
            BB1(gamma_idx,sig_idx) = mean_testerror_1;
            BB2(gamma_idx,sig_idx) = mean_testerror_2;
            BB3(gamma_idx,sig_idx) = mean_testerror_3;
        end
        
    end
    
    %% ================  state x_1 ================
    
    [minBB1, idx] = min(BB1(:));
    [p, q] = ind2sub(size(BB1),idx);
    gamma = gamma_range(p);
    sig = sigma_range(q);
    K = KernelMatrix(t,'RBF_kernel', sig);
    m = size(K,1);
    A = [K + (1/gamma) * eye(m), ones(m,1);...
        ones(m,1)' ,0];
    B1 = [y1;0];
    result1 =  A\B1;
    alpha1 = result1(1:m);
    b1 = result1(end);
    tnew = t0:(sampling_time/2):tf;
    tnew = tnew';
    Knew1 = KernelMatrix(t,'RBF_kernel', sig, tnew);
    xxt1 = t*ones(1,size(tnew,1));
    xxt2 = tnew*ones(1,size(t,1));
    coft = 2*(xxt1-xxt2')/sig;
    Kyt1 = coft.* Knew1;
    
    %% ================  state x_2 ================
    
    [minBB2, idx] = min(BB2(:));
    [p, q] = ind2sub(size(BB2),idx);
    gamma = gamma_range(p);
    sig = sigma_range(q);
    K = KernelMatrix(t,'RBF_kernel', sig);
    A = [K + (1/gamma) * eye(m), ones(m,1);...
        ones(m,1)' ,0];
    B2 = [y2;0];
    result2 =  A\B2;
    alpha2 = result2(1:end-1);
    b2 = result2(end);
    Knew2 = KernelMatrix(t,'RBF_kernel', sig, tnew);
    xxt1 = t*ones(1,size(tnew,1));
    xxt2 = tnew*ones(1,size(t,1));
    coft = 2*(xxt1-xxt2')/sig;
    Kyt2 = coft.* Knew2;
    
    
    %% ================  state x_3 ================

    [minBB3, idx] = min(BB3(:));
    [p, q] = ind2sub(size(BB3),idx);
    gamma = gamma_range(p);
    sig = sigma_range(q);
    K = KernelMatrix(t,'RBF_kernel', sig);
    A = [K + (1/gamma) * eye(m), ones(m,1);...
        ones(m,1)' ,0];
    B3 = [y3;0];
    result3 =  A\B3;
    alpha3 = result3(1:end-1);
    b3 = result3(end);
    Knew3 = KernelMatrix(t,'RBF_kernel', sig, tnew);
    xxt1 = t*ones(1,size(tnew,1));
    xxt2 = tnew*ones(1,size(t,1));
    coft = 2*(xxt1-xxt2')/sig;
    Kyt3 = coft.* Knew3;
      
    
    
    yhat1 = Knew1' * alpha1  + b1;
    yhat2 = Knew2' * alpha2  + b2;
    yhat3 = Knew3' * alpha3  + b3;
    yphat1 = (Kyt1' )* alpha1 ;
    yphat2 = (Kyt2' )* alpha2 ;
    yphat3 = (Kyt3' )* alpha3 ;
    

    C1 = [(yhat2-yhat1)' ; zeros(1,size(yhat1,1)) ; zeros(1,size(yhat1,1))];
    A1 = reshape(C1,[],1);
    C2 = [ zeros(1,size(yhat2,1)) ; yhat1';  zeros(1,size(yhat2,1))];
    A2 = reshape(C2,[],1);
    C3 = [ zeros(1,size(yhat3,1)) ; zeros(1,size(yhat3,1)) ; -yhat3' ];
    A3 = reshape(C3,[],1);
    C4 = [yphat1' ; yphat2' + (yhat2'+yhat1'.*yhat3') ; yphat3' - yhat1'.*yhat2'];
    A4 = reshape(C4,[],1);
    
    A = [A1 A2 A3];
    B = A4;
    theta_hat = A\B;
    
    
    erpara1 = abs(ExactTheta(1)- theta_hat(1));
    erpara2 = abs(ExactTheta(2)-theta_hat(2));
    erpara3 = abs(ExactTheta(3)-theta_hat(3));
    
    fprintf('estimated theta1= %f => abs_error=%e \n', theta_hat(1),erpara1 )
    fprintf('estimated theta2= %f => abs_error=%e \n', theta_hat(2), erpara2)
    fprintf('estimated theta3= %f =>  abs_error=%e \n\n', theta_hat(3),erpara3 )
    
    Par1(itr) = theta_hat(1);
    Par2(itr) = theta_hat(2);
    Par3(itr) = theta_hat(3);
end
cprintf( [1 0.1 0],'**** Estimated parameters of the Lorenz system ***** \n');

par1 = mean(Par1);
par2 = mean(Par2);
par3 = mean(Par3);

fprintf( 'final estimated theta_1= %f, \n', par1)
fprintf( 'final estimated theta_2= %f, \n', par2)
fprintf( 'final estimated theta_3= %f, \n\n', par3)

Theta_hat = [par1;par2;par3];
Error = abs(ExactTheta-Theta_hat);
sol = ode45(@ridg,[t0 tf],initial,options,Theta_hat);
extsol = ode45(@ridg,[t0 tf],initial,options,ExactTheta);
Yhat = deval(sol,tnew)';
Y = deval(extsol,tnew)';


%% ===========  plotting each of the state ===============

figure
set(gcf,'Position',[250 250 1500 700])
subplot(3,2,1)
plot(tnew,Y(:,1),'k','linewidth',2)
hold on,grid minor
plot(tnew,Yhat(:,1),'r--','linewidth',2)
% plot(t,y1,'o', 'MarkerFaceColor',[0 0 0])
legend('True z','Estimated z')
xlabel('t','FontSize', 24)
ylabel('x_1(t)','FontSize', 24)

subplot(3,2,3)
plot(tnew,Y(:,2),'k','linewidth',2)
hold on,grid minor
plot(tnew,Yhat(:,2),'r--','linewidth',2)
% plot(t,y2,'o', 'MarkerFaceColor',[0 0 0])
legend('True z','Estimated z')
xlabel('t','FontSize', 24)
ylabel('x_2(t)','FontSize', 24)

subplot(3,2,5)
plot(tnew,Y(:,3),'k','linewidth',2)
hold all, grid minor
plot(tnew,Yhat(:,3),'r--','linewidth',2)
hold all
% plot(t,y3,'k*')
legend('True z','Estimated z')
xlabel('t','FontSize', 24)
ylabel('x_3(t)','FontSize', 24)

% ===========  plotting phase-space ===============

subplot(3,2,[2,4,6])
plot3(Y(:,1),Y(:,2),Y(:,3),'k','linewidth',2)
hold on
plot3(Yhat(:,1),Yhat(:,2),Yhat(:,3),'r--','linewidth',1.5)
grid on
xlabel('x_1(t)','FontSize', 24)
ylabel('x_2(t)','FontSize', 24)
zlabel('x_3(t)','FontSize', 24)
title('Phase-Space','FontSize', 24 )
legend('Real','Estimated')
