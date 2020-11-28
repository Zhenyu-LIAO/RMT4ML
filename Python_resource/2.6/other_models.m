%% Section 2.6: Information-plus-noise, deformed Wigner, and other models
% This page contains simulations in Section 2.6.

%% Haar sample covariance (Theorem 2.16)
% Generate a (Gaussian i.i.d.) random matrix $Z$ of dimension $p \times n$
% and a Haar random matrix $U = Z (Z^T Z)^{-\frac12}$ from Z
% Generate the associated data matrix $X = C^{\frac12} U$
close all; clear; clc

coeff = 2;
p = 800*coeff;
n = 600*coeff;

eigs_C = [ones(p/4,1); 2*ones(p/4,1); 3*ones(p/2,1)];
% eigs_C = [ones(p/2,1); 3*ones(p/2,1)];
C = diag(eigs_C); % population covariance
% C = toeplitz((.4).^(0:(p-1)));
% eigs_C = eig(C);

v = [-ones(p/2,1);ones(p/2,1)]/sqrt(p);

C = C + 10*v*(v');
eigs_C = eig(C);

% Z=randn(p,2*p);
% C = Z*Z'/(2*p);
% eigs_C = eig(C,'vector');

% Z = randn(p,n);
% U = Z*sqrtm(inv(Z'*Z));
% %U = Z*((Z'*Z)^(-1/2));
% %U = U(:,1:n);
% X = sqrtm(C)*U;

Z = randn(p);
U = sqrtm(inv(Z*Z'))*Z(:,1:n);
X= sqrtm(C)*U;


% index= 100;
% u = U(:,index);
% Ui = U(:,[1:index-1,index+1:end]);
% Pi = eye(p) - U*(U');
% 
% 
% lambda = .5;
% Q = inv(p/n*sqrtm(C)*U*(U')*sqrtm(C)+lambda*eye(p));
% Qi = inv(p/n*sqrtm(C)*Ui*(Ui')*sqrtm(C)+lambda*eye(p));
% 
% delta = u'*sqrtm(C)*Qi*sqrtm(C)*u;
% eta = trace(Q*C)/p;



% (1-c)*delta - (eta - c*delta/(1+delta/c))

% clc
% M = 1/(1+ (1/c - 1)*delta );
% trace(C*inv(C*M +lambda*eye(p)))/p
% eta
% alpha = trace(Q)/p;
% 
% trace(C*inv( (1 - lambda*alpha)/delta*C + lambda*eye(p) ))/p
% 
% alpha
% 
% trace(inv( (1 - lambda*alpha)/delta*C + lambda*eye(p) ))/p


% Empirical eigenvalues of the Haar sample covariance $\frac{p}n X X^T = \frac{p}n C^{\frac12} U U^T C ^{\frac12}$
% versus the solution of fixed-point equation in Theorem 2.16
%SCM = p/n*(X')*(X);
SCM = p/n*(X)*(X');

%a= ones(p,1)/sqrt(p);
a = v;
zs = linspace(-3,-1,20);
store_emp = zeros(length(zs),2);
store_theo = zeros(length(zs),2);
for j=1:length(zs)
    z = zs(j);
    
    Q = inv(SCM - z*eye(p));
    tm=1;
    tm_tmp=0;
    while abs(tm-tm_tmp)>1e-6
        tm_tmp=tm;
        tm = 1/( - z + 1/n*sum(eigs_C./(1+tm*eigs_C))*(1+z*tm*n/p)); %/(1-n/p)
    end
    bar_Q = -inv(eye(p) + tm*C)/z;
    
    store_emp(j,1) = trace(Q)/p;
    store_theo(j,1) = trace(bar_Q)/p;
    store_emp(j,2) = a'*Q*a;
    store_theo(j,2) = a'*bar_Q*a;
    z
end


figure
subplot(1,2,1)
hold on
plot(store_emp(:,1),'*')
plot(store_theo(:,1),'x')

subplot(1,2,2)
hold on
plot(store_emp(:,2),'*')
plot(store_theo(:,2),'x')
%% Romain's code
clear; close all; clc

p=1200;n=600;
eigs_C = [ones(p/4,1); 3*ones(p/4,1); 5*ones(p/2,1)];
%eigs_C = [ones(p/2,1); 3*ones(p/2,1)];
C = diag(eigs_C);
%Z=randn(p,2*p);
%C=Z*Z'/(2*p);

Z=randn(p);
W=inv(Z*Z')^.5*Z(:,1:n);
X=C^.5*W;

clear i
zs = linspace(.5,1.5,5)+1i*1e-3;
for z = zs
    Q=inv(p/n*X*X'-z*eye(p));
    %tQ=inv(p/n*X'*X-z*eye(n));
    
    tm=1;tm_tmp=0;
    while abs(tm-tm_tmp)>1e-6
        tm_tmp=tm;
        tm=1/(-z+1/n*trace(C*inv(tm*C+eye(p)))*(1+z*n/p*tm));
    end
    (1/p*trace(Q) - 1/p*trace(-1/z*inv(eye(p)+C*tm)))/(1/p*trace(-1/z*inv(eye(p)+C*tm)))
    z
end


%A=toeplitz(.5.^(0:p-1));
A = eye(p);
1/p*trace(A*Q),1/p*trace(A*(-1/z*inv(eye(p)+C*tm)))

%a=ones(p,1)/sqrt(p);a'*Q*a,a'*(-1/z*inv(eye(p)+C*tm))*a
