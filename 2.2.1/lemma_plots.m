%% Section 2.2.1: Key lemmas and identities
% This page contains simulations in Section 2.2.1.

%% Illustration of Lemma 2.9
clear; close all; clc; 

coeff = 2;
p = 200*coeff;
n = 500*coeff;

X = randn(p,n);

rng(928);
M = X*X'/n;
u = ones(p,1)/sqrt(p);
tau = 1;

M_plus = M + tau*(u*u');

[V_M, D_M] = eig(M);
d_M_plus = eig(M_plus);

eig_index = 100;
eig1 = D_M(eig_index,eig_index);
eig2 = D_M(eig_index+1,eig_index+1);
eig3 = D_M(eig_index+2,eig_index+2);


Tol = 1e-5;
func = @(lambda) tau*u'*( ( M - lambda*eye(p) )\u );
lambda_range_12 = linspace(eig1+Tol, eig2-Tol,100);
lambda_range_23 = linspace(eig2+Tol, eig3-Tol/2,100);


figure
hold on
p1 = plot(eig1,0,'bo');
plot(eig2,0,'bo')
plot(eig3,0,'bo')
p2 = plot(d_M_plus(eig_index),0,'rx');
plot(d_M_plus(eig_index+1),0,'rx')
p3 = plot(lambda_range_12, arrayfun(func,lambda_range_12),'m');
plot(lambda_range_23, arrayfun(func,lambda_range_23),'m')
yline(-1,'--k');
xline(eig1,'--k');
xline(eig2,'--k');
xline(eig3,'--k');

xline(d_M_plus(eig_index),'--r');
xline(d_M_plus(eig_index+1),'--r');
axis([eig1-50*Tol eig3+50*Tol -8 8])
set(gca,'xtick',[])
xlabel('Eigenvalues ($z$)', 'Interpreter', 'latex')
ylabel('$\tau u^T Q_M(z) u$', 'Interpreter', 'latex')
legend([p1 p2 p3], '$\lambda(M)$', '$\lambda(M+ \tau uu^T)$', '$\tau u^T Q_M(z) u$', 'Interpreter', 'latex', 'FontSize', 15, 'Location','northwest')


