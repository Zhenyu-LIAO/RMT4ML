%% Section 2.2.3: Large dimensional sample covariance matrices and deformed semi-circles
% This page contains simulations in Section 2.2.3.

%% Large dimensional sample covariance matrix (Theorem 2.5)
% Generate a (Gaussian i.i.d.) random matrix $Z$ of dimension $p \times n$
% Generate the associated data matrix $X = C^{\frac12} Z$
close all; clear; clc

coeff = 3;
p = 100*coeff;
n = 1000*coeff;
c = p/n;

eigs_C = [ones(p/3,1); 3*ones(p/3,1); 5*ones(p/3,1)];
% fell free to change to eigs_C = [1,..,3,...,5...]
C = diag(eigs_C); % population covariance

Z = randn(p,n);
X = C^(1/2)*Z;
%%
% Empirical eigenvalues of the sample covariance matrix $\frac1n X X^T = \frac1n C^{\frac12} Z Z^T C ^{\frac12}$
% versus the solution of fixed-point equation in Theorem 2.5
SCM = X*(X')/n;
eigs_SCM = eig(SCM);
edges=linspace(min(eigs_SCM)-.1,max(eigs_SCM)+.2,60);

clear i % make sure i stands for the imaginary unit
y = 1e-5;
zs = edges+y*1i;
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
histogram(eigs_SCM,edges, 'Normalization', 'pdf');
hold on;
plot(edges,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Theorem 2.5', 'FontSize', 15)

%% The bi-correlated model (Theorem 2.6)
% Generate a (Gaussian i.i.d.) random matrix $Z$ of dimension $p \times n$
% Generate the associated data matrix $X = C^{\frac12} Z \tilde C^{\frac12}$
close all; clear; clc

coeff = 3;
p = 200*coeff;
n = 1000*coeff;
c = p/n;

eigs_C = [ones(p/3,1); 3*ones(p/3,1); 8*ones(p/3,1)];
eigs_tilde_C = [ones(n/2,1); 2*ones(n/2,1)];
% fell free to vary the setting of eigs_C and eigs_tilde_C
C = diag(eigs_C);
tilde_C = diag(eigs_tilde_C);

Z = randn(p,n);
X = C^(1/2)*Z*tilde_C^(1/2);

%%
% Empirical eigenvalues of the sample covariance matrix $\frac1n X X^T = \frac1n C^{\frac12} Z \tilde C Z^T C^{\frac12}$
% versus the solution of (symmetric) fixed-point equation systems in Theorem 2.6
SCM = X*(X')/n;
eigs_SCM = eig(SCM);
edges=linspace(min(eigs_SCM)-.1,max(eigs_SCM)+.1,60);

clear i % make sure i stands for the imaginary unit
y = 1e-5;
zs = edges+y*1i;
mu = zeros(length(zs),1);

delta = [0,0]; % corresponds to [delta, delta_delta] in Theorem 2.6
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
histogram(eigs_SCM,edges, 'Normalization', 'pdf');
hold on;
plot(edges,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Theorem 2.6', 'FontSize', 15)

%% Sample covariance of $k$-class mixture models (Theorem 2.7)
% Generate a (Gaussian i.i.d.) random matrix $Z$ of dimension $p \times n$
% Generate the associated data matrix $X = [C_1^{\frac12}z_1, \ldots, C_k^{\frac12}z_i,\ldots]$
close all; clear; clc

coeff = 3;
p = 200*coeff;
n = 1000*coeff;
c = p/n;
k = 3; % three classes in total

eigs_C = @(a) [ones(p/3,1); a*ones(p/3,1); 1/a*ones(p/3,1)];
C = @(a) diag(eigs_C(a));
% fell free to vary the setting of C_a, a=1,...,k

cs  = ones(k,1)/k; % the vector of c_a, a=1,...,k, proportion in each class
if length(cs) ~= k
    error('Error: number of classes mismatches!')
end

X=zeros(p,n);
for i=1:k
    X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=C(i)^(1/2)*randn(p,cs(i)*n);
end

%%
% Empirical eigenvalues of the mixture sample covariance matrix $\frac1n X X^T$
% versus the solution of the system of equations in Theorem 2.7
SCM = X*(X')/n;
eigs_SCM = eig(SCM);
edges=linspace(min(eigs_SCM)-.1,max(eigs_SCM)+.1,60);

clear i % make sure i stands for the imaginary unit
y = 1e-5;
zs = edges+y*1i;
mu = zeros(length(zs),1);

tilde_g = ones(k,1); % corresponds to [tilde_g_1, ..., tilde_g_k] in Theorem 2.6
for j = 1:length(zs)
    z = zs(j);
    
    tilde_g_tmp = zeros(k,1);
    g = ones(k,1);
    %watch_dog = 1; % to avoid possible numerical convergence issue
    while max(abs(tilde_g-tilde_g_tmp))>1e-6 %&& watch_dog<50
        tilde_g_tmp = tilde_g;
        
        eigs_C_sum = zeros(p,1);
        for b = 1:k
            eigs_C_sum = eigs_C_sum + cs(b)*tilde_g_tmp(b)*eigs_C(b);
        end
        
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
histogram(eigs_SCM,edges, 'Normalization', 'pdf');
hold on;
plot(edges,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Theorem 2.7', 'FontSize', 15)

%% The deformed semi-circle law (Theorem 2.8)
% Generate a (Gaussian) symmetric random matrix $Z$ of size $n \times n$.
close all; clear; clc
coeff = 2;
n=500*coeff;

Z=randn(n);
Z_U = triu(Z);
X = triu(Z) + triu(Z)'-diag(diag(triu(Z)));

bern_mask_p = .1;
% fell free to change the probability of success
bern_mask = (rand(n,n)<bern_mask_p);
bern_mask_U = triu(bern_mask);
bern_mask = triu(bern_mask_U) + triu(bern_mask_U)'-diag(diag(triu(bern_mask_U)));
%%
% Empirical eigenvalues of $\frac1{\sqrt n} X.*bern_mask$ versus the deformed semi-circle law.
DSC = (X.*bern_mask)/sqrt(n);
eigs_DSC = eig(DSC);
edges=linspace(min(eigs_DSC)-.1,max(eigs_DSC)+.1,60);

clear i % make sure i stands for the imaginary unit
y = 1e-5;
zs = edges+y*1i;
mu = zeros(length(zs),1);

% g_vec = zeros(n,1);
% for index = 1:length(zs)
%     z = zs(index);
%     
%     g_vec_tmp = ones(n,1);
%     while max(abs( g_vec - g_vec_tmp )) > 1e-6
%         g_vec = g_vec_tmp;
%         for j = 1:n
%             g_vec(j) = -sum( bern_mask(j,:)./(1+g_vec)' )/n/z/z;
%         end
%     end
%    m = -sum(1./(1+g_vec))/n/z;
%    mu(index) = imag(m)/pi;
% end

g = 0;
for j=1:length(zs)
    z = zs(j);
    
    g_tmp = 1;
    while abs(g - g_tmp)>1e-6
        g_tmp=g;
        g = -bern_mask_p/(1+g)/z/z;
    end
    m = -1/(1+g)/z;
    mu(j)=imag(m)/pi;
end

figure
histogram(eigs_DSC,edges, 'Normalization', 'pdf');
hold on;
plot(edges,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Theorem 2.8', 'FontSize', 15)
