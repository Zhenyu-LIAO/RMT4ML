%% Section 3.1.1: GLRT asymptotics
% This page contains simulations in Section 3.1.1
% Detection of the presence of statistical information from white noise
close all; clear; clc

p = 100;
n = 300;
c = p/n;

a = [ones(p/2,1); -ones(p/2,1)]; %%% "determnistic" data structure
a = a/norm(a);
sigma = 1;

nb_average_loop = 5000;
f_alpha_loop = (1+sqrt(c))^2+linspace(-5,5,100)*n^(-2/3);
emp_type_1_error = zeros(size(f_alpha_loop));
theo_type_1_error = zeros(size(f_alpha_loop));


for i = 1:length(f_alpha_loop)
    f_alpha = f_alpha_loop(i); %%% decision thredhold
    
    T = @(X) norm(X*(X')/n)/( trace(X*(X')/n)/p);
    
    tmp_error = 0;
    for average_loop = 1:nb_average_loop
        s = randn(n,1); %%% random signal
        X = sigma*randn(p,n);
        tmp_error = tmp_error + (T(X)< f_alpha);
    end
    emp_type_1_error(i) = tmp_error/nb_average_loop;
    [~,theo_type_1_error(i)] = tracy_widom_appx((f_alpha - (1+sqrt(c))^2)*(1+sqrt(c))^(-4/3)*c^(1/6)*n^(2/3), 1);
end

figure
hold on
plot(f_alpha_loop,emp_type_1_error)
plot(f_alpha_loop,theo_type_1_error)


close all; clear; clc

coeff = 4;
p = 600*coeff;
n = 1000*coeff;
%cs = [1/4 3/4];
cs = [1/2 1/2];
k = 2;

eigs_C = @(l) [ones(p/3,1); l*ones(p/3,1); l^2*ones(p/3,1)];
C= @(l) diag(eigs_C(l));
means = @(l) (-1)^l*[ones(p/2,1); -ones(p/2,1)]/sqrt(p);


Z1 = randn(p,n*cs(1));
Z2 = randn(p,n*cs(2));
X = [sqrtm(C(1))*Z1, sqrtm(C(2))*Z2];
SCM = X*(X')/n;

% eigs_SCM = eig(SCM);
% edges=linspace(min(eigs_SCM)-.1,max(eigs_SCM)+.1,60);
%
% clear i % make sure i stands for the imaginary unit
% y = 1e-5;
% zs = edges+y*1i;
% mu = zeros(length(zs),1);
%
% tilde_g = ones(k,1); % corresponds to [tilde_g_1, ..., tilde_g_k] in Theorem 2.6
% for j = 1:length(zs)
%     z = zs(j);
%
%     tilde_g_tmp = zeros(k,1);
%     %%watch_dog = 1;
%     while min(abs(tilde_g-tilde_g_tmp))>1e-6 %%&& watch_dog<50
%         tilde_g_tmp = tilde_g;
%
%         eigs_C_sum = zeros(p,1);
%         for b = 1:k
%             eigs_C_sum = eigs_C_sum + cs(b)*tilde_g(b)*eigs_C(b);
%         end
%
%         g = ones(k,1);
%         for a = 1:k
%             g(a) = -1/n/z*sum( eigs_C(a)./(1 + eigs_C_sum) );
%             tilde_g(a) = -1/z/(1+g(a));
%         end
%         %%watch_dog = watch_dog + 1;
%     end
%
%     eigs_C_sum = zeros(p,1);
%     for b = 1:k
%         eigs_C_sum = eigs_C_sum + cs(b)*tilde_g_tmp(b)*eigs_C(b);
%     end
%     m = -1/p/z*sum(1./(1 + eigs_C_sum) );
%     mu(j)=imag(m)/pi;
%     z
% end
%
% figure
% histogram(eigs_SCM, edges, 'Normalization', 'pdf');
% hold on;
% plot(edges,mu,'r', 'Linewidth',2);
% legend('Empirical eigenvalues', 'Theorem 2.7', 'FontSize', 15)


z = -.1;

Q_c = inv( SCM - z*eye(p) );
U = [means(1), means(2), sqrtm(C(1))*Z1*ones(n*cs(1),1)/(n*cs(1)), sqrtm(C(2))*Z2*ones(n*cs(2),1)/(n*cs(2))];

Delta = zeros(4,4);
Delta(3,3) = cs(1);
Delta(4,4) = cs(1);
Q = inv( SCM - U*Delta*(U') - z*eye(p) );


tilde_g = ones(k,1);
tilde_g_tmp = zeros(k,1);
%%watch_dog = 1;
while min(abs(tilde_g-tilde_g_tmp))>1e-6 %%&& watch_dog<50
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
    %%watch_dog = watch_dog + 1;
end


eigs_C_sum = zeros(p,p);
for b = 1:k
    eigs_C_sum = eigs_C_sum + cs(b)*tilde_g(b)*C(b);
end

bar_Q_c = -inv( eigs_C_sum + eye(p) )/z;

gamma = -z;

T1 = [1 -1 -1 -1]*(U')*Q*U*[1; -1; 1; -1]/2
T2 = [-1 1 -1 -1]*(U')*Q*U*[1; -1; 1; -1]/2

disp('Romain')
( (means(1) - means(2))'*bar_Q_c*(means(1) - means(2)) -(1-gamma*tilde_g(1))^2/(cs(1))^2 + (1-gamma*tilde_g(2))^2/(cs(2))^2 )/2
( -(means(1) - means(2))'*bar_Q_c*(means(1) - means(2)) -(1-gamma*tilde_g(1))^2/(cs(1))^2 + (1-gamma*tilde_g(2))^2/(cs(2))^2 )/2

disp('Mine')
( (means(1) - means(2))'*bar_Q_c*(means(1) - means(2)) -(1-gamma*tilde_g(1))/(cs(1)*gamma*tilde_g(1)) + (1-gamma*tilde_g(2))/(cs(2)*gamma*tilde_g(2)) )/2
( -(means(1) - means(2))'*bar_Q_c*(means(1) - means(2)) -(1-gamma*tilde_g(1))/(cs(1)*gamma*tilde_g(1)) + (1-gamma*tilde_g(2))/(cs(2)*gamma*tilde_g(2)) )/2
%
% U'*Q*U
%
% A = [ means(1)'*bar_Q*means(1), means(1)'*bar_Q*means(2); means(2)'*bar_Q*means(1), means(2)'*bar_Q*means(2) ]
% B = [ (1 + z*tilde_g(1))/(cs(1)), 0; 0, (1 +z*tilde_g(2))/(cs(2)) ]
%

%% FUNCTION
function [pdftwappx, cdftwappx] = tracy_widom_appx(x, i)
%
% [pdftwappx, cdftwappx]=tracywidom_appx(x, i)
%
% SHIFTED GAMMA APPROXIMATION FOR THE TRACY-WIDOM LAWS, by M. Chiani, 2014
% code publicly available https://www.mathworks.com/matlabcentral/fileexchange/44711-approximation-for-the-tracy-widom-laws
%
% TW ~ Gamma[k,theta]-alpha
%
% [pdf,cdf]=tracywidom_appx(x,i) for i=1,2,4 gives TW1, TW2, TW4
%

kappx = [46.44604884387787, 79.6594870666346, 0, 146.0206131050228];   %  K, THETA, ALPHA
thetaappx = [0.18605402228279347, 0.10103655775856243, 0, 0.05954454047933292];
alphaappx = [9.848007781128567, 9.819607173436484, 0, 11.00161520109004];

cdftwappx = cdfgamma(x+alphaappx(i), thetaappx(i), kappx(i));

pdftwappx = pdfgamma(x+alphaappx(i), thetaappx(i), kappx(i));

end

function pdf=pdfgamma(x, ta, ka)
if(x > 0)
    pdf=1/(gamma(ka)*ta^ka) * x.^(ka - 1) .* exp(-x/ta);
else
    pdf=0 ;
end
end

function cdf=cdfgamma(x, ta, ka)
if(x > 0)
    cdf=gammainc(x/ta,ka);
else
    cdf=0;
end

end
