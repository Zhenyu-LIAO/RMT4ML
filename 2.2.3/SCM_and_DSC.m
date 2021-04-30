%% Section 2.2.3: Large dimensional sample covariance matrices and generalized semicircles
% This page contains simulations in Section 2.2.3.

%% Large dimensional sample covariance matrix (Theorem 2.6)
% Generate a (Gaussian i.i.d.) random matrix $Z$ of dimension $p \times n$
% Generate the associated data matrix $X = C^{\frac12} Z$
close all; clear; clc

coeff = 3;
p = 100*coeff;  
n = 1000*coeff;
c = p/n;

eigs_C = [ones(p/3,1); 3*ones(p/3,1); 7*ones(p/3,1)]; %eigs_C = [1,..,3,...,5...]
C = diag(eigs_C); % population covariance

Z = randn(p,n);
X = sqrtm(C)*Z;
%%
% Empirical eigenvalues of the sample covariance matrix $\frac1n X X^T = \frac1n C^{\frac12} Z Z^T C ^{\frac12}$
% versus the solution of fixed-point equation in Theorem 2.6
Tol = 1e-1;
SCM = X*(X')/n;
eigs_SCM = eig(SCM);
edges_mu=linspace(min(eigs_SCM)-Tol,max(eigs_SCM)+Tol,200);

clear i % make sure i stands for the imaginary unit
y = 1e-5;
zs = edges_mu+y*1i;
mu = zeros(length(zs),1);

tilde_m=0;
for j=1:length(zs)
    z = zs(j);
    
    tilde_m_tmp=-1;
    while abs(tilde_m-tilde_m_tmp)>1e-6
        tilde_m_tmp=tilde_m;
        tilde_m = 1/( -z + 1/n*sum(eigs_C./(1+tilde_m*eigs_C)) );
    end
    
    m = tilde_m/c+(1-c)/(c*z);
    mu(j)=imag(m)/pi;
end

figure
histogram(eigs_SCM, 40, 'Normalization', 'pdf','EdgeColor', 'white');
hold on;
plot(edges_mu,mu,'r', 'Linewidth',2);
legend('Empirical spectrum', 'Theorem 2.6', 'Interpreter', 'latex', 'FontSize', 15);

%% The bi-correlated model (Theorem 2.7)
% Generate a (Gaussian i.i.d.) random matrix $Z$ of dimension $p \times n$
% Generate the associated data matrix $X = C^{\frac12} Z \tilde C^{\frac12}$
close all; clear; clc

coeff = 3;
p = 200*coeff;
n = 1000*coeff;

eigs_C = [ones(p/3,1); 3*ones(p/3,1); 8*ones(p/3,1)];
eigs_tilde_C = [ones(n/2,1); 3/2*ones(n/2,1)];

C = diag(eigs_C);
tilde_C = diag(eigs_tilde_C);

Z = randn(p,n);
X = sqrtm(C)*Z*sqrtm(tilde_C);

%%
% Empirical eigenvalues of the sample covariance matrix $\frac1n X X^T = \frac1n C^{\frac12} Z \tilde C Z^T C^{\frac12}$
% versus the solution of (symmetric) fixed-point equation systems in
% Theorem 2.7
Tol = 1e-1;
SCM = X*(X')/n;
eigs_SCM = eig(SCM);
edges_mu=linspace(min(eigs_SCM)-Tol,max(eigs_SCM)+Tol,100);

clear i % make sure i stands for the imaginary unit
y = 1e-5;
zs = edges_mu+y*1i;
mu = zeros(length(zs),1);

delta = [0,0]; % corresponds to [delta, delta_delta] in Theorem 2.7
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

figure
histogram(eigs_SCM, 40, 'Normalization', 'pdf', 'EdgeColor', 'white');
hold on;
plot(edges_mu,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Theorem 2.7', 'Interpreter', 'latex', 'FontSize', 15);

%% Sample covariance of $k$-class mixture models (Theorem 2.8)
% Generate a (Gaussian i.i.d.) random matrix $Z$ of dimension $p \times n$
% Generate the associated data matrix $X = [C_1^{\frac12}z_1, \ldots, C_a^{\frac12}z_i,\ldots]$
close all; clear; clc

coeff = 3;
p = 200*coeff;
n = 1000*coeff;
c = p/n;
k = 3; % three classes in total

eigs_C = @(a) [ones(p/3,1); a*ones(p/3,1); a^2*ones(p/3,1)];
C = @(a) diag(eigs_C(a));

%cs  = ones(k,1)/k; % the vector of c_a, a=1,...,k, proportion in each class
cs = [1/4 1/4 1/2];

if length(cs) ~= k
    error('Error: number of classes mismatches!')
end

X=zeros(p,n);
for i=1:k
    X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=sqrtm(C(i))*randn(p,cs(i)*n);
end

%%
% Empirical eigenvalues of the mixture sample covariance matrix $\frac1n X X^T$
% versus the solution of the system of equations in Theorem 2.8
Tol = 1e-1;
SCM = X*(X')/n;
eigs_SCM = eig(SCM);
edges_mu=linspace(min(eigs_SCM)-Tol,max(eigs_SCM)+Tol,100);

clear i % make sure i stands for the imaginary unit
y = 1e-5;
zs = edges_mu+y*1i;
mu = zeros(length(zs),1);

tilde_g = ones(k,1); % corresponds to [tilde_g_1, ..., tilde_g_k] in Theorem 2.8
for j = 1:length(zs)
    z = zs(j);
    
    tilde_g_tmp = zeros(k,1);
    %watch_dog = 1; % to avoid possible numerical convergence issue
    while max(abs(tilde_g-tilde_g_tmp))>1e-6 %&& watch_dog<50
        tilde_g_tmp = tilde_g;
        
        eigs_C_sum = zeros(p,1);
        for b = 1:k
            eigs_C_sum = eigs_C_sum + cs(b)*tilde_g(b)*eigs_C(b);
        end
        
        g = ones(k,1);
        for a = 1:k
            g(a) = -1/n/z*sum( eigs_C(a)./(1 + eigs_C_sum) );
            tilde_g(a) = -1/z/(1+g(a));
        end
    end
    
    eigs_C_sum = zeros(p,1);
    for b = 1:k
        eigs_C_sum = eigs_C_sum + cs(b)*tilde_g_tmp(b)*eigs_C(b);
    end
    m = -1/p/z*sum(1./(1 + eigs_C_sum) );
    mu(j)=imag(m)/pi;
end

figure
histogram(eigs_SCM, 40, 'Normalization', 'pdf', 'EdgeColor', 'white');
hold on;
plot(edges_mu,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Theorem 2.8', 'Interpreter', 'latex', 'FontSize', 15);

%% The generalized semicircle law (Theorem 2.9)
% Generate a (Gaussian) symmetric random matrix $Z$ of size $n \times n$.
close all; clear; clc
coeff = 2;
n=500*coeff;

Z=randn(n);
Z_U = triu(Z);
X = triu(Z) + triu(Z)'-diag(diag(triu(Z)));

bern_mask_p = .5; 

bern_mask = (rand(n,n)<bern_mask_p);
bern_mask_U = triu(bern_mask);
bern_mask = triu(bern_mask_U) + triu(bern_mask_U)'-diag(diag(triu(bern_mask_U)));

%%
% Empirical eigenvalues of $\frac1{\sqrt n} X.*Mask$ versus the generalized semicircle law.
DSC = (X.*bern_mask)/sqrt(n);
Tol = 1e-1;
eigs_DSC = eig(DSC);
edges_mu=linspace(min(eigs_DSC)-Tol,max(eigs_DSC)+Tol,60);

clear i % make sure i stands for the imaginary unit
y = 1e-5;
zs = edges_mu+y*1i;
mu = zeros(length(zs),1);

g = 0;
for j=1:length(zs)
    z = zs(j);
    
    g_tmp = 1;
    while abs(g - g_tmp)>1e-6
        g_tmp=g;
        g = -bern_mask_p/(z + g);
    end
    m = -1/(z+g);
    mu(j)=imag(m)/pi;
end


figure
histogram(eigs_DSC, 30, 'Normalization', 'pdf', 'EdgeColor', 'white');
hold on;
plot(edges_mu,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Theorem 2.9', 'Interpreter', 'latex', 'FontSize', 15);
