%% Section 2.6: Information-plus-noise, deformed Wigner, and other models
% This page contains simulations in Section 2.6.

%% Haar sample covariance (Theorem 2.16)
% Generate a (Gaussian i.i.d.) random matrix $Z$ of dimension $p \times n$
% and a Haar random matrix $U = Z (Z^T Z)^{-\frac12}$ from Z
% Generate the associated data matrix $X = C^{\frac12} U$
close all; clear; clc

coeff = 2;
p = 900*coeff;
n = 300*coeff;

%eigs_C = [ones(p/4,1); 3*ones(p/4,1); 5*ones(p/2,1)];
%eigs_C = [ones(p/2,1); 3*ones(p/2,1)];
%C = diag(eigs_C); % population covariance
Z=randn(p,2*p);
C = Z*Z'/(2*p);
eigs_C = eig(C,'vector');

% Z = randn(p,n);
% U = Z*sqrtm(inv(Z'*Z));
% %U = Z*((Z'*Z)^(-1/2));
% %U = U(:,1:n);
% X = sqrtm(C)*U;

Z = randn(p);
U = inv(Z*Z')^.5*Z(:,1:n);
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
eigs_SCM = eig(SCM);

%eigs(SCM)
edges=linspace(min(eigs_SCM)+.1,max(eigs_SCM)+.2,60);

%edges = linspace(1,20,100);
% z = -.5; my version, not good in a DE sense
% Q = inv(SCM - z*eye(p));
% trace(Q)/p
% 
% m = 1;
% m_tmp = 0;
% while abs(m-m_tmp)>1e-6
%     m_tmp = m;
%     
%     %tmp = p/n*( (p-n)/n + n/p +z*delta )/( 1 + delta*z );
%     tmp = ( p*(p-n) + n^2 + z*p^2*m )/n/(n+z*p*m);
%     m = mean( 1./( tmp*eigs_C - z) );
% end
% tmp = ( p*(p-n) + n^2 + z*p^2*m )/n/(n+z*p*m);
% bar_Q = inv( tmp*C - z*eye(p) );
% trace(bar_Q)/p


clear i % make sure i stands for the imaginary unit
y = 1e-4;
zs = edges+y*1i;
mu = zeros(length(zs),1);

%%% my version 
% m = 1;
% for j=1:length(zs)
%     z = zs(j);    
%     m_tmp = 0;
%     while abs(m-m_tmp)>1e-6
%         m_tmp = m;
%         
%         %tmp = p/n*( (p-n)/n + n/p +z*delta )/( 1 + delta*z );
%         tmp = ( p*(p-n) + n^2 + z*p^2*m )/n/(n+z*p*m);
%         m = mean( 1./( tmp*eigs_C - z) );
%     end
%     
%     %tm = m*p/n - (n-p)/n/z;
%     mu(j)=imag(m)/pi;
%     z
% end



for j=1:length(zs)
    z = zs(j);
    
    watch_dog = 1;
    tm=1;tm_tmp=0;
    while abs(tm-tm_tmp)>1e-5 && watch_dog < 50
        watch_dog = watch_dog + 1;
        tm_tmp=tm;
        %tm=1/(-z+1/n*trace(C*inv(tm*C+eye(p)))*(1+z*n/p*tm));
        tm = 1/( - z + 1/n*sum(eigs_C./(1+tm*eigs_C))*(1+z*n/p*tm) );
    end
    %mu(j) = imag( -1/z*mean( 1./(tm*eigs_C+1)))/pi;
    mu(j) = imag( -1/z/p*trace( inv(tm*C+eye(p))))/pi;
    z
%     Q = inv(p/n*X*X'-z*eye(p));
%     bar_Q = -1/z*inv(eye(p) + tm*C);
%     z
%     (trace(Q)/p - trace(bar_Q)/p)/(trace(bar_Q)/p)
end
    
    


figure
histogram(real(eigs_SCM), 30, 'Normalization', 'pdf');
hold on;
plot(edges,mu,'r', 'Linewidth',2);
%plot(edges,mu_SCM,'b', 'Linewidth',2);
legend('Empirical eigenvalues', 'Theorem 2.16', 'FontSize', 15)

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
end


%%
%A=toeplitz(.5.^(0:p-1));
A = eye(p);
1/p*trace(A*Q),1/p*trace(A*(-1/z*inv(eye(p)+C*tm)))

%a=ones(p,1)/sqrt(p);a'*Q*a,a'*(-1/z*inv(eye(p)+C*tm))*a
