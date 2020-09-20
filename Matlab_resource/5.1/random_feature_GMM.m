%% Section 5.1.2: Delving deeper into limiting kernel
% This page contains simulations in Section 5.1.2.

%% Basic settings
close all; clear; clc

testcase='GMM'; % among 'GMM', 'MNIST' and 'fashion-MNIST', 'kannada-MNIST'
testcase_option = 'mixed'; % among 'iid', 'mean', 'cov', 'orth' and 'mixed'
sigma_fun = 't';  
% among mean: 't', 'sign', 'posit', 'erf', 'sin'
% covariance: 'cos', 'abs', 'exp'
% balance: 'ReLU', 'poly2'

coeff = 1;
n = 256*coeff;
n_test = 1000*coeff;
cs = [1/4 1/4 1/4 1/4];
k = length(cs);

switch testcase
    case 'GMM'
        p = 512*coeff;
        
        switch testcase_option % a = 1,2,3,4
            case 'means'
                means = @(a) [zeros(a-1,1);1;zeros(p-a,1)]*3;
                covs  = @(a) eye(p);
            case 'cov'
                means = @(a) zeros(p,1);
                covs  = @(a) eye(p)*(1+(a-1)/sqrt(p)*15);
            case 'orth'
                means = @(a) zeros(p,1);
                covs = @(a) toeplitz((4*(a-1)/10).^(0:(p-1)));
            case 'mixed'
                %means = @(a) [zeros(a-1,1);1;zeros(p-a,1)]*5;
                %covs  = @(a) eye(p)*(1+(a-1)/sqrt(p)*15);   
                % four-class setting
                means = @(a) [a<=2;a>=3;zeros(p-2,1)]*5;
                covs  = @(a) eye(p)*(1+or(a==2,a==4)/sqrt(p)*15);
        end
    case 'MNIST'
        init_data = loadMNISTImages('./../../datasets/MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('./../../datasets/MNIST/train-labels-idx1-ubyte');
    case 'fashion-MNIST'
        init_data = loadMNISTImages('./../../datasets/fashion-MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('./../../datasets/fashion-MNIST/train-labels-idx1-ubyte');
    case 'kannada-MNIST'
        init_data = loadMNISTImages('./../../datasets/kannada-MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('./../../datasets/kannada-MNIST/train-labels-idx1-ubyte');
end

switch testcase % real-world data pre-processing
    case {'MNIST', 'fashion-MNIST','kannada-MNIST'}
        [labels,idx_init_labels]=sort(init_labels,'ascend');
        data=init_data(:,idx_init_labels);
        
        init_n=length(data(1,:));
        p=length(data(:,1));
        
        selected_labels=[1 2];
        
        data = data/max(data(:));
        mean_data=mean(data,2);
        norm2_data=0;
        for i=1:init_n
            norm2_data=norm2_data+1/init_n*norm(data(:,i)-mean_data)^2;
        end
        data=(data-mean_data*ones(1,size(data,2)))/sqrt(norm2_data)*sqrt(p);
        
        selected_data = cell(k,1);
        cascade_selected_data=[];
        j=1;
        for i=selected_labels
            selected_data{j}=data(:,labels==i);
            cascade_selected_data = [cascade_selected_data, selected_data{j}];
            j = j+1;
        end
        
        % recentering of the k classes
        mean_selected_data=mean(cascade_selected_data,2);
        norm2_selected_data=mean(sum(abs(cascade_selected_data-mean_selected_data*ones(1,size(cascade_selected_data,2))).^2));
        
        for j=1:length(selected_labels)
            selected_data{j}=(selected_data{j}-mean_selected_data*ones(1,size(selected_data{j},2)))/sqrt(norm2_selected_data)*sqrt(p);
        end
        
        means = @(l) mean(selected_data{l+1},2);
        covs = @(l) 1/length(selected_data{l+1})*(selected_data{l+1}*selected_data{l+1}')-means(l)*means(l)';
end

switch testcase % generate data
    case {'MNIST', 'fashion-MNIST','kannada-MNIST'}
        X=zeros(p,n);
        X_test=zeros(p,n_test);
        y=zeros(n,1);
        y_test=zeros(n_test,1);
        for i=1:k % random data picking
            data = selected_data{i}(:,randperm(size(selected_data{i},2)));
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=data(:,1:n*cs(i));
            X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n+1:n+n_test*cs(i));
            
            y(sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n) = (-1)^i*ones(cs(i)*n,1);
            y_test(sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test) = (-1)^i*ones(cs(i)*n_test,1);
        end
    case 'GMM'
        W=zeros(p,n);
        W_test=zeros(p,n_test);
        y=zeros(n,1);
        y_test=zeros(n_test,1);
        for i=1:k
            W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=sqrt(covs(i))*randn(p,cs(i)*n);
            W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=sqrt(covs(i))*randn(p,cs(i)*n_test);
        end
        
        X=zeros(p,n);
        X_test=zeros(p,n_test);
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)+means(i)*ones(1,cs(i)*n);
            X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)+means(i)*ones(1,cs(i)*n_test);
            
            y(sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n) = (-1)^i*ones(cs(i)*n,1);
            y_test(sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test) = (-1)^i*ones(cs(i)*n_test,1);
        end
end

X = X/sqrt(p); % data normalization to ensure operator norm boundedness
X_test = X_test/sqrt(p);

switch sigma_fun
    case 't'
        sig = @(t) t;
        K_xy = @(x,y) x'*y;
        
    case 'poly2'
        poly2A = 1; poly2B = 1;poly2C = 1;
        sig = @(t) poly2A*t.^2+poly2B*t+poly2C;
        K_xy = @(x,y) poly2A^2*( 2*(x'*y).^2+(x.^2)'*ones(size(x,1))*(y.^2))+poly2B^2*(x'*y)+poly2A*poly2C*(diag(x'*x)*ones(1,size(y,2))+ones(size(x,2),1)*diag(y'*y)')+poly2C^2;
        
    case 'ReLU'
        sig = @(t) max(t,0);
        angle_xy = @(x,y) diag(1./sqrt(diag(x'*x)))*(x'*y)*diag(1./sqrt(diag(y'*y)));
        K_xy = @(x,y) sqrt(diag(x'*x))*sqrt(diag(y'*y))'/(2*pi).*(angle_xy(x,y).*acos(-angle_xy(x,y))+sqrt(1-angle_xy(x,y).^2));
        
    case 'sign'
        sig = @(t) sign(t);
        K_xy = @(x,y) 2/pi*asin(diag(1./sqrt(diag(x'*x)))*(x'*y)*diag(1./sqrt(diag(y'*y))));
        
    case 'posit'
        sig = @(t) (sign(t)+1)/2;
        K_xy = @(x,y) 1/2-1/(2*pi)*acos(diag(1./sqrt(diag(x'*x)))*(x'*y)*diag(1./sqrt(diag(y'*y))));
        
    case 'erf'
        sig = @(t) erf(t);
        K_xy = @(x,y) 2/pi*asin(diag(1./sqrt(1+2*diag(x'*x)))*(2*x'*y)*diag(1./sqrt(1+2*diag(y'*y))));
        
    case 'cos'
        sig = @(t) cos(t);
        K_xy = @(x,y) diag(exp(-diag(x'*x/2)))*cosh(x'*y)*diag(exp(-diag(y'*y/2)'));
        
    case 'sin'
        sig = @(t) sin(t);
        K_xy = @(x,y) diag(exp(-diag(x'*x/2)))*sinh(x'*y)*diag(exp(-diag(y'*y/2)'));
        
    case 'abs'
        sig = @(t) abs(t);
        angle_xy = @(x,y) diag(1./sqrt(diag(x'*x)))*(x'*y)*diag(1./sqrt(diag(y'*y)));
        K_xy = @(x,y) 2*sqrt(diag(x'*x))*sqrt(diag(y'*y))'/pi.*(angle_xy(x,y).*(acos(-angle_xy(x,y))-pi/2)+sqrt(1-angle_xy(x,y).^2));
    
    case 'exp'
        sig = @(t) exp(-t.^2/2);
        K_xy = @(x,y) 1./sqrt( 1 + (x.^2)'*ones(size(x,1))*(y.^2) + diag(x'*x)*ones(1,size(y,2))+ones(size(x,2),1)*diag(y'*y)' - (x'*y).^2);
end

P = eye(n) - ones(n,n)/n;

%% Top eigenvectors of recentered kernel matrices

% N = 2048;
% W = randn(N,p);
% Sigma = sig(W*X);
% G = Sigma'*Sigma/N;
% 
% G = real(G);
% PGP = P*G*P;
% [U_PGP,L_PGP]=svd(PGP);
% 
% figure % Top eigenvectors of the (empirical) $PGP$
% subplot(2,1,1)
% plot(U_PGP(:,1));
% title('First eigenvalues of $PGP$', 'Interpreter', 'latex')
% subplot(2,1,2)
% plot(U_PGP(:,2));
% title('Second eigenvalues of $PGP$', 'Interpreter', 'latex')


K = real(K_xy(X,X));
PKP = P*K*P;
[U_PKP,L_PKP]=svd(PKP);

figure % Top eigenvectors of $PKP$ 
subplot(2,1,1)
plot(U_PKP(:,1));
title('First eigenvalues of $PKP$', 'Interpreter', 'latex')
subplot(2,1,2)
plot(U_PKP(:,2));
title('Second eigenvalues of $PKP$', 'Interpreter', 'latex')

