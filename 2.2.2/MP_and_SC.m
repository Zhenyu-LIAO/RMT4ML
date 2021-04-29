%% Section 2.2.2: The Marcenko-Pastur and semicircle laws
% This page contains simulations in Section 2.2.2.

%% The Marcenko-Pastur law (Theorem 2.4）
% Generate a (Gaussian) random matrix $X$ of dimension $p \times n$.
close all; clear; clc

coeff = 5; 
p = 100*coeff;
n = 10000*coeff;
c = p/n;

X = randn(p,n);
%%
% Plot the empirical eigenvalues of $\hat C = \frac1n X X^T$ versus the (limiting) Marcenko-Pastur law.
SCM = X*(X')/n;
a = (1-sqrt(c))^2;
b = (1+sqrt(c))^2;
edges=linspace(a-eps,b+eps,60);

figure
histogram(eig(SCM),30, 'Normalization', 'pdf', 'EdgeColor', 'white');
hold on;
mu=sqrt( max(edges-a,0).*max(b-edges,0) )/2/pi/c./edges;
plot(edges,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Marcenko-Pastur law', 'Interpreter', 'latex', 'FontSize', 15)
xlabel('Eigenvalues of $\hat C$', 'Interpreter', 'latex', 'FontSize', 15)
ylabel('Histogram', 'Interpreter', 'latex', 'FontSize', 15)
axis([0.7 1.3 0 4.5]);

%% The Wigner semicircle law（Theorem 2.5)
% Generate a (Gaussian) *symmetric* random matrix $X$ of size $n \times n$.
close all; clear; clc

coeff = 5;
n=200*coeff;

Z=randn(n);
X = triu(Z) + triu(Z)'-diag(diag(triu(Z)));
%%
% Plot the empirical eigenvalues of $\frac{X}{\sqrt{n}}$ versus the limiting semicircle law.
edges=linspace(-2-eps,2+eps,50);

figure
histogram(eig(X/sqrt(n)),30,'Normalization','pdf', 'EdgeColor', 'white');
hold on;
mu = sqrt( max(4 - edges.^2,0) )/2/pi;
plot(edges,mu,'r','LineWidth',2);
legend('Empirical eigenvalues', 'Wigner semi-circle law', 'Interpreter', 'latex', 'FontSize', 15);
axis([-2.5 2.5 0 0.45]);
