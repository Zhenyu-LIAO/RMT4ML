%% Section 2.2.2: The Marcenko-Pastur and semi-circle laws
% This page contains simulations in Section 2.2.2.

%% The Marcenko-Pastur law (Theorem 2.3）
% Generate a (Gaussian) random matrix $X$ of dimension $p \times n$.
close all; clear; clc

coeff = 5; 
p = 100*coeff;
n = 1000*coeff;
c = p/n;

X = randn(p,n);
%%
% Empirical eigenvalues of $\frac1n X X^T$ versus the Marcenko-Pastur law.
SCM = X*(X')/n;
a = (1-sqrt(c))^2;
b = (1+sqrt(c))^2;
edges=linspace(a-eps,b+eps,60);

figure
histogram(eig(SCM),edges, 'Normalization', 'pdf');
hold on;
mu=sqrt( max(edges-a,0).*max(b-edges,0) )/2/pi/c./edges;
plot(edges,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Marcenko-Pastur law', 'FontSize', 15)

%% The Wigner semi-circle law （Theorem 2.4)
% Generate a (Gaussian) symmetric random matrix $X$ of size $n \times n$.

coeff = 5;
n=100*coeff;

Z=randn(n);
Z_U = triu(Z);
X = triu(Z) + triu(Z)'-diag(diag(triu(Z)));
%%
% Empirical eigenvalues of $\frac{X}{\sqrt{n}}$ versus the semi-circle law.
edges=linspace(-2-eps,2+eps,50);

figure
histogram(eig(X/sqrt(n)),edges,'Normalization','pdf');
hold on;
mu = sqrt( max(4 - edges.^2,0) )/2/pi;
plot(edges,mu,'r','LineWidth',2);
legend('Empirical eigenvalues', 'Wigner semi-circle law', 'FontSize', 15)
