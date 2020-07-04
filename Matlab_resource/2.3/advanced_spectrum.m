%% Section 2.3: Advanced spectrum considerations for sample covariances
% This page contains simulations in Section 2.3.

%% Section 2.3.1 Limiting spectrum (part 1): Theorem 2.9
% Study of the support of (the limiting spectrum of) sample covariance matrix $\frac1n C^{\frac12} Z Z^T C ^{\frac12}$
% as well as its connection to the functional inverse $x(\tilde m)$
close all; clear; clc

coeff = 3;
p = 100*coeff;
n = 1000*coeff;
c = p/n;

eig_C = [1,3,7];
cs = [1/3 1/3 1/3];
eigs_C = [eig_C(1)*ones(p/3,1); eig_C(2)*ones(p/3,1); eig_C(3)*ones(p/3,1)];
C = diag(eigs_C); % population covariance

Z = randn(p,n);
X = C^(1/2)*Z;

SCM = X*(X')/n; %%% sample covariance matrix and its empirical spectral measure
eigs_SCM = eig(SCM);
edges=linspace(min(eigs_SCM)-.1,max(eigs_SCM)+.1,100);

clear i
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

figure %%% limiting versus empirical spectral measure of SCM
hold on
histogram(eigs_SCM,edges, 'Normalization', 'pdf');
plot(edges,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Limiting spectrum', 'FontSize', 15)

% functional inverse
%x = @(tilde_m, eigs_C) -1./tilde_m + c*( eigs_C(1)./(1+eigs_C(1)*tilde_m)/3 + eigs_C(2)./(1+eigs_C(2)*tilde_m)/3 + eigs_C(3)./(1+eigs_C(3)*tilde_m)/3 );
% calling SCM_func_inv(tilde_m, eigs_C, cs, c) defined at the end of script
x = @(tilde_m) SCM_func_inv(tilde_m, eig_C, cs, c);

tilde_ms = linspace(-2,1,1000);
for lambda = [eig_C,0]
    tol = eps;
    tilde_ms(tilde_ms<=-1/lambda+tol & tilde_ms>=-1/lambda-tol)=NaN;
end
figure %%% corresponds to Figure 2.4
hold on
p1 = plot(tilde_ms, x(tilde_ms), 'r');
p2 = xline(-1/eig_C(1),'--k');
xline(-1/eig_C(2),'--k');
xline(-1/eig_C(3),'--k');
p3 = plot(zeros(p,1),eigs_SCM,'xb');
yline(0,'k');
xline(0,'k');
axis([-2 1 -2 12]) %%% set different axis limits, to see for instance when tilde_m >0
xlabel('$\tilde m$', 'Interpreter', 'latex');
ylabel('$x(\tilde m)$', 'Interpreter', 'latex');
legend([p1 p2 p3], {'$x(\tilde m)$', '$-\frac1{\tilde m} \in supp(\nu)$', 'empirical eigenvalues of SCM'},...
    'Interpreter', 'latex', 'Location', 'northwest', 'FontSize', 15);

%% Section 2.3.1 Limiting spectrum (part 2): variable change to relate $supp(\nu)$ and $supp(\mu)$
%
% Study of the function $\gamma(\cdot)$ that maps $z(\tilde m)$ to $-\frac1{\tilde m}$
% and in particular, the exclusion region that cannot be reached by $\gamma$
close all; clear; clc

coeff = 3;
p = 100*coeff;
n = 1000*coeff;
% p = 200*coeff;
% n = 100*coeff;
c = p/n;

eigs_C = [ones(p/3,1); 3*ones(p/3,1); 5*ones(p/3,1)];
C = diag(eigs_C);
Z = randn(p,n);
X = C^(1/2)*Z;
SCM = X*(X')/n;
eigs_SCM = eig(SCM);

clear i
y_min = -1;
y_max = 1;
x_min = -1;
x_max = 10;
% x_max = 30;

zs1 = (x_max:-0.1:x_min) + y_max*1i;
zs2 = x_min + (y_max:-0.1:y_min)*1i;
zs3 = (x_min:0.1:x_max) + y_min*1i;
zs4 = x_max + (y_min:0.1:y_max)*1i;

zs = [zs1, zs2, zs3, zs4]; %%% contour Gamma_mu circling around the (limiting) support mu
gamma_zs = zeros(length(zs),1);

tilde_m=0;
for j=1:length(zs)
    z = zs(j);
    
    tilde_m_tmp=-1;
    while abs(tilde_m-tilde_m_tmp)>1e-6
        tilde_m_tmp=tilde_m;
        tilde_m = 1/( -z + 1/n*sum(eigs_C./(1+tilde_m*eigs_C)) );
    end
    
    gamma_zs(j)= -1/tilde_m;
end

figure
subplot(2,1,1)
hold on
plot(zs,'r','Linewidth',2)
plot(eigs_SCM,zeros(p,1),'xk')
yline(0,'k');
xline(0,'k');
axis([-2 11 -2 2])
% axis([-2 30 -2 2])
xlabel('$\Re[z]$', 'Interpreter', 'latex')
ylabel('$\Im[z]$', 'Interpreter', 'latex')
legend('Typical contour $\Gamma_\mu$ (of $z$)', 'Empirical spectrum of SCM', 'Interpreter', 'latex')
subplot(2,1,2)
hold on
plot(gamma_zs, 'r', 'Linewidth',2)
plot(eigs_C,zeros(p,1),'xk');
yline(0,'k');
xline(0,'k');
axis([-2 11 -2 2])
% axis([-5 25 -6 6])
xlabel('$\Re[-1/\tilde m(z)]$', 'Interpreter', 'latex')
ylabel('$\Im[-1/\tilde m(z)]$', 'Interpreter', 'latex')
legend('Typical contour $\Gamma_\nu$ (of $-1/\tilde m(z)$)', 'Support of $\nu$' , 'Interpreter', 'latex')

%% Section 2.3.2 "No eigenvalue outside the support" (Theorem 2.10)
% Study the behavior of SCM eigenvalues that possibly "escapes" from the
% limiting support $\mu$
close all; clear; clc

coeff = 6;
p = 100*coeff;
n = 1000*coeff;
c = p/n;

eig_C = [1,3,7];
eigs_C = [eig_C(1)*ones(p/3,1); eig_C(2)*ones(p/3,1); eig_C(3)*ones(p/3,1)];
C = diag(eigs_C); %%% population covariance
nu_student = 3; %%% degrees of freedom nu of Student's t distribution

Z1 = randn(p,n);
Z2 = trnd(nu_student,p,n)/sqrt(nu_student/(nu_student-2));

X1 = C^(1/2)*Z1;
X2 = C^(1/2)*Z2;

SCM1 = X1*(X1')/n; %%% Gaussian SCM
SCM2 = X2*(X2')/n; %%% Student's t SCM
eigs_SCM1 = eig(SCM1);
eigs_SCM2 = eig(SCM2);
edges1=linspace(min(eigs_SCM1)-.1,max(eigs_SCM1)+.2,100);
edges2=linspace(min(eigs_SCM2)-.1,max(eigs_SCM2)+.2,100);

clear i
y = 1e-5;
zs = edges1+y*1i;
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
subplot(2,1,1)
hold on
histogram(eigs_SCM1,edges1, 'Normalization', 'pdf');
plot(edges1,mu,'r', 'Linewidth',2);
title('Gaussian SCM')
legend('Empirical eigenvalues', 'Limiting spectrum', 'FontSize', 12)
subplot(2,1,2)
hold on
histogram(eigs_SCM2,edges2, 'Normalization', 'pdf');
plot(edges1,mu,'r', 'Linewidth',2);
legend('Empirical eigenvalues', 'Limiting spectrum', 'FontSize', 12)
title('Student-t SCM')

%% FUNCTIONS
function [x,x_d] = SCM_func_inv(tilde_m, eig_C, cs, c)
%SCM_func_inv functional inverse of Stieltjes transform of large sample
%covariance model
%   INPUT: Stieltjes transform tilde_m, (k-discrete) eigenvalues of C (or
%   nu), vector cs=p_a/p for a=1,...k, ratio c=p/n
%   OUTPUT: functional inverse x (of tilde_m) and its first derivative x_d

if length(eig_C) ~= length(cs)
    error('Error: nb of (discrete) eigenvalues and nb of classes not equal!')
end

x = -1./tilde_m;
x_d = 1./(tilde_m.^2);
for a=1:length(cs)
    x = x + c*cs(a)*eig_C(a)./(1+eig_C(a)*tilde_m);
    x_d = x_d - c*cs(a)*eig_C(a)^2./(1+eig_C(a)*tilde_m).^2;
end

end
