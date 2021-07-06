%% Section 3.3: M-estimators of scatter
% This page contains simulations in Section 3.3.

%% Robust estimator of scatter versus the classical sample covariance matrix
% Generate a random matrix $Z \in R^{p \times n}$, having columns uniform
% on the $\sqrt{p}$-sphere, and i.i.d. random $\Gamma(0.5,2)$ vector $\tau$.
close all; clear; clc

coeff = 5;
p = 100*coeff;
n = 500*coeff;
c = p/n;

eigs_C = [ones(p/4,1); 3*ones(p/4,1); 10*ones(p/2,1)];
C = diag(eigs_C); % population covariance
tau = gamrnd(0.5,2,n,1);
eigs_tilde_C = tau;

Z = randn(p,n);
inv_norm_Z = p./diag(Z'*Z);
Z = Z*diag(sqrt(inv_norm_Z)); % uniformly distribution on the sphere
X = sqrtm(C)*Z*diag(sqrt(tau));

SCM = X*(X')/n;
eigs_SCM = eig(SCM);
edges=linspace(min(eigs_SCM)*0.9,max(eigs_SCM)*1.1,500);

clear i
y = 1e-5;
zs = edges+y*1i;
mu = zeros(length(zs),1);

delta = [0,0];
for j = 1:length(zs)
    z = zs(j);
    
    delta_tmp = [1,1];
    %watch_dog = 1; % to avoid possible numerical convergence issue
    while max(abs(delta-delta_tmp))>1e-6 %&& watch_dog < 50
        delta_tmp = delta;
        delta(1) = -1/n/z*sum(eigs_C./( 1 + delta_tmp(2)*eigs_C ));
        delta(2) = -1/n/z*sum(eigs_tilde_C./( 1 + delta_tmp(1)*eigs_tilde_C ));
    end
    
    m = -1/p/z*sum(1./(1 + delta(2)*eigs_C) );
    mu(j)=imag(m)/pi;
end

figure(1)
histogram(eigs_SCM, 50, 'Normalization', 'pdf','EdgeColor', 'white');
hold on;
plot(edges,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues of SCM', 'Limiting law', 'Interpreter', 'latex', 'FontSize', 15)

%% Asymptotic equivalent for the robust estimator of scatter $\hat C$
alpha = 0.2;
u = @(x) (1+alpha)./(alpha+x);
phi = @(x) x.*u(x);
g = @(x) x./(1-c*phi(x));

hat_C_tmp = eye(p);
hat_C = eye(p)/2;
while norm(hat_C -  hat_C_tmp)/norm(hat_C)>1e-3
    hat_C_tmp = hat_C;
    inv_hat_C_X = hat_C\X;
    hat_C = X*diag(u(diag(X'*inv_hat_C_X/p)))*(X')/n;
end

gamma = 1;
gamma_tmp = 0;
while abs(gamma-gamma_tmp)>1e-6
    gamma_tmp = gamma;
    
    syms x
    g_inv = finverse(g(x));
    x = gamma*tau;
    v = u(eval(g_inv));
    gamma = 1/(sum( tau.*v./(1+c*gamma*tau.*v) )/n);
end

hat_S = X*diag(v)*(X')/n;
eigs_hat_S = eig(hat_S);

eigs_hat_C = eig(hat_C);
edges=linspace(min(eigs_hat_C)*0.9,max(eigs_hat_C)*1.1,500);


eigs_tilde_C = tau.*v;

clear i 
y = 1e-5;
zs = edges+y*1i;
mu = zeros(length(zs),1);

delta = [0,0];
for j = 1:length(zs)
    z = zs(j);
    
    delta_tmp = [1,1];
    %watch_dog = 1; % to avoid possible numerical convergence issue
    while max(abs(delta-delta_tmp))>1e-6 %&& watch_dog < 50
        delta_tmp = delta;
        delta(1) = -1/n/z*sum(eigs_C./( 1 + delta_tmp(2)*eigs_C ));
        delta(2) = -1/n/z*sum(eigs_tilde_C./( 1 + delta_tmp(1)*eigs_tilde_C ));
    end
    
    m = -1/p/z*sum(1./(1 + delta(2)*eigs_C) );
    mu(j)=imag(m)/pi;
end


figure(2)
histogram(eigs_hat_C, 50, 'Normalization', 'pdf', 'EdgeColor', 'white');
hold on;
plot(edges,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues of $\hat C$', 'Limiting law $\mu_{\hat C}$', 'FontSize', 15, 'Interpreter', 'latex')


figure(3)
histogram(eigs_hat_S, 50, 'Normalization', 'pdf', 'EdgeColor', 'white');
hold on;
plot(edges,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues of $\hat S$', 'Limiting law $\mu_{\hat S}$', 'FontSize', 15, 'Interpreter', 'latex')

%% Robust spiked model
close all; clear; clc

coeff = 2;
p = 128*coeff;
n = 512*coeff;
c = p/n;


nu_student = 100; %%% degrees of freedom nu of Student's t distribution
t = trnd(nu_student,n,1)/sqrt(nu_student/(nu_student-2));
tau = t.^2;

a = [-ones(p/2,1);ones(p/2,1)]/sqrt(p);
s = randn(n,1)*.65;

Z = randn(p,n);
inv_norm_Z = p./diag(Z'*Z);
Z = Z*diag(sqrt(inv_norm_Z)); % uniformly distribution on the sphere
X = a*(s') + Z*diag(sqrt(tau));

alpha = .2;
u = @(x) (1+alpha)./(alpha+x);
phi = @(x) x.*u(x);
g = @(x) x./(1-c*phi(x));

hat_C_tmp = eye(p);
hat_C = eye(p)/2;
while norm(hat_C -  hat_C_tmp)/norm(hat_C)>1e-3
    hat_C_tmp = hat_C;
    inv_hat_C_X = hat_C\X;
    hat_C = X*diag(u(diag(X'*inv_hat_C_X/p)))*(X')/n;
end

eigs_hat_C = eig(hat_C);
edges=linspace(min(eigs_hat_C)*0.9,max(eigs_hat_C)*1.1,300);

gamma = 1;
gamma_tmp = 0;
while abs(gamma-gamma_tmp)>1e-6
    gamma_tmp = gamma;
    
    syms x
    g_inv = finverse(g(x));
    x = gamma*tau;
    v = u(eval(g_inv));
    gamma = 1/(sum( tau.*v./(1+c*gamma*tau.*v) )/n);
end

eigs_C = ones(p,1);
eigs_tilde_C = tau.*v;

clear i % make sure i stands for the imaginary unit
y = 1e-5;
zs = edges+y*1i;
mu = zeros(length(zs),1);

delta = [0,0]; 
for j = 1:length(zs)
    z = zs(j);
    
    delta_tmp = [1,1];
    while max(abs(delta-delta_tmp))>1e-6 
        delta_tmp = delta;
        delta(1) = -1/n/z*sum(eigs_C./( 1 + delta_tmp(2)*eigs_C ));
        delta(2) = -1/n/z*sum(eigs_tilde_C./( 1 + delta_tmp(1)*eigs_tilde_C ));
    end
    
    m = -1/p/z*sum(1./(1 + delta(2)*eigs_C) );
    mu(j)=imag(m)/pi;
end

S_plus = (1+alpha)/(1-c*(1+alpha))*(1+sqrt(c))^2/gamma;

figure
histogram(eigs_hat_C, 50, 'Normalization', 'pdf', 'EdgeColor', 'white');
hold on;
plot(edges,mu,'r', 'Linewidth',2);
xline(S_plus,'--');
legend('Empirical eigenvalues of $\hat C$', 'Limiting law', 'FontSize', 15, 'Interpreter', 'latex')
