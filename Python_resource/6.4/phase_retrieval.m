%% Section 6.4: Generalized linear classifier
% This page contains simulations in Section 6.1.

%% Histogram of $\beta_{-i}^T \tilde x_i$ versus the limiting Gaussian behavior
close all; clear; clc

close all; clear; clc

coeff = 2;
p = 100*coeff;
n = 1500*coeff;
c = p/n;

compute_expectation = 'integral'; % 'empirical' or 'integral'

% processing/truncating function
%f = @(t) 0*(t>=10)+t;
f = @(t) (max(t,0)-1)./(max(t,0)+sqrt(2/c)-1);

value_posit = 100;
alpha = [zeros(value_posit-1,1); 1; zeros(p-value_posit,1)]; %%% vector to recover
%alpha = [-ones(p/2,1);ones(p/2,1)];
alpha = alpha/norm(alpha);


X = randn(p,n);
v = X'*alpha;
y = v.^2;
fD = diag(f(y));
eigs_fD = diag(fD);

X_perp = X - alpha*(alpha')*X;

chi_s = @(t) exp(-t/2)./sqrt(t)/sqrt(2)/gamma(1/2);

H = X*fD*(X')/n;
[V_H,eigs_H] = eig(H);
H_perp = X_perp*fD*(X_perp')/n;

eigs_H = diag(eigs_H);

edges = linspace(min(eigs_H)*0.9, max(eigs_H)*1.1, 200);

clear i % make sure i stands for the imaginary unit
y = 1e-5;
zs = edges+y*1i;
dens = zeros(length(zs),1);

m=1;
for j=1:length(zs)
    z = zs(j);
    
    m_tmp=-1;
    while abs(m-m_tmp)>1e-6
        m_tmp=m;
        m = 1/( -z + mean(eigs_fD./(1+c*m*eigs_fD)));
    end
    dens(j)=imag(m)/pi;
    z
end


figure
histogram(eigs_H, 30, 'Normalization', 'pdf')
hold on
plot(edges,dens)

top_eig_vec = V_H(:,1);
if top_eig_vec'*alpha<0
    top_eig_vec = -top_eig_vec;
end

figure
plot(top_eig_vec,'x')
hold on 
plot(alpha)

%% spikes
spike_search_range = linspace(max(eigs_H)*.8,max(eigs_H)*1.1,500);
det_store = zeros(length(spike_search_range),1);

for i = 1:length(det_store)
    x = spike_search_range(i);
    
    m=1;
    m_tmp=-1;
    while abs(m-m_tmp)>1e-6
        m_tmp=m;
        m = 1/( -x + mean(eigs_fD./(1+c*m*eigs_fD)));
    end
    
    %det_store(i) = mean( eigs_D.^2./( 1 + c*m*eigs_D) ) + 1/m;
    %det_store(i) = mean( eigs_D.^2./( 1 + c*m*eigs_D) ) - x; % good
    %det_store(i) = mean( eigs_D.*(v.^2)./( 1 + c*m*eigs_D) ) - x; % general
    det_store(i) = mean( eigs_fD.*(v.^2)./( 1 + c*m*eigs_fD) ) - x; % general
    
    %trapz_y = (trapz_x.^4 + trapz_x.^2).*exp(-trapz_x.^2/2)/sqrt(2*pi)./(1+trapz_x.^2*c*m);
    %det_store = trapz(trapz_x,trapz_y) - 1;
    x
end

figure
hold on
plot(spike_search_range, det_store)
plot(max(eigs_H),0,'x')
yline(0,'--k');
%%
alpha'*Q*alpha

s = mean( (v.^2-1).*eigs_fD./( 1 + c*m*eigs_fD ) );

b = 1/m;
%bar_Q = m*eye(p) + alpha*(alpha')*m/(s - z);
%bar_Q = m*eye(p) - m*s/(s+1/m)*alpha*(alpha');
%bar_Q = inv( s*alpha*(alpha') + b*eye(p) );
bar_Q = m*(eye(p) - alpha*(alpha')) - alpha*(alpha')/z;

alpha'*bar_Q*alpha


trace(Q)/p
trace(bar_Q)/p

%%
x_m_func = @(m) -c./m + integral( @(t) sqrt(t).*exp(-t/2)./(1+m.*t)/sqrt(2)/gamma(1/2), eps, 50);
range_m = linspace(-5,1,1000);

x_m = zeros(size(range_m));
for i=1:length(range_m)
    m = range_m(i);
    x_m(i) = x_m_func(m);
end
plot(range_m, x_m)
