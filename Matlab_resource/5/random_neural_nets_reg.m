%% Section 5.1.1: Regression with random neural networks
% This page contains simulations in Section 5.1.1.

%% Basic settings
close all; clear; clc

testcase='fashion-MNIST'; % among 'GMM', 'MNIST' and 'fashion-MNIST', 'kannada-MNIST'
testcase_option = 'mixed'; % among 'iid', 'mean', 'cov', 'orth' and 'mixed'
sigma_fun = 'exp';  % among 'ReLU', 'sign', 'posit', 'erf', 'poly2', 'cos','sin','abs', 'exp'

coeff = 1;
n = 1000*coeff;
n_test = 1000*coeff;
cs = [1/2 1/2];
k = length(cs);

switch sigma_fun
    case 'poly2'  % A.x^2 + B.x + C
        poly2A=-1/2;
        poly2B=0;
        poly2C=1;
        
        W_choice = 'gauss'; % among 'bern', 'skewed_bern', 'gauss', 'student'
        nu_student = 7;
        
    otherwise
        W_choice = 'gauss';
end

switch testcase
    case 'GMM'
        p = 512*coeff;
        
        switch testcase_option % l = 0 or 1
            case 'means'
                means = @(l) [zeros(l+1,1);1;zeros(p-l-2,1)]*3;
                covs  = @(l) eye(p);
            case 'cov'
                means = @(l) zeros(p,1);
                covs  = @(l) eye(p)*(1+l/sqrt(p)*15);
            case 'orth'
                means = @(l) zeros(p,1);
                covs = @(l) toeplitz((4*l/10).^(0:(p-1)));
            case 'mixed'
                means = @(l) [zeros(l+1,1);1;zeros(p-l-2,1)]*5;
                covs  = @(l) eye(p)*(1+l/sqrt(p)*20);     
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
            W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=sqrt(covs(i-1))*randn(p,cs(i)*n);
            W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=sqrt(covs(i-1))*randn(p,cs(i)*n_test);
        end
        
        X=zeros(p,n);
        X_test=zeros(p,n_test);
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)+means(i-1)*ones(1,cs(i)*n);
            X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)+means(i-1)*ones(1,cs(i)*n_test);
            
            y(sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n) = (-1)^i*ones(cs(i)*n,1);
            y_test(sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test) = (-1)^i*ones(cs(i)*n_test,1);
        end
end

X = X/sqrt(p); % data normalization to ensure operator norm boundedness
X_test = X_test/sqrt(p);

switch W_choice
    case 'gauss'
        m2 = 1;
        m3 = 0;
        m4 = 3;
        
    case 'bern'
        m2 = 1;
        m3 = 0;
        m4 = 1;
        
    case 'skewed_bern'
        m2 = 1;
        m3 = -2/sqrt(3);
        m4 = 7/3;
        
    case 'student'
        m2 = 1;
        m3 = 0;
        m4 = 6/(nu_student-4)+3;
end

switch sigma_fun
    case 't'
        sig = @(t) t;
        K_xy = @(x,y) m2*(x'*y);
        
    case 'poly2'
        sig = @(t) poly2A*t.^2+poly2B*t+poly2C;
        K_xy = @(x,y) poly2A^2*(m2^2*(2*(x'*y).^2+(x.^2)'*ones(size(x,1))*(y.^2))+(m4-3*m2^2)*(x.^2)'*(y.^2))+poly2B^2*(m2*(x'*y))+poly2A*poly2B*m3*((x.^2)'*y+x'*(y.^2))+poly2A*poly2C*m2*(diag(x'*x)*ones(1,size(y,2))+ones(size(x,2),1)*diag(y'*y)')+poly2C^2;
        
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

%% Theoretical evaluation via Corollary 8
tic

K_X = real(K_xy(X,X));
[U_K_X,L_K_X]=svd(K_X);
U_K_X = real(U_K_X);
eig_K_X = diag(L_K_X);
Up_K_X = U_K_X'*K_X;

U_K_y = U_K_X'*y;

K_XXtest = real(K_xy(X,X_test));
U_K_XXtest = U_K_X'*K_XXtest;
D_U_K_XXtest_2 = diag(U_K_XXtest*U_K_XXtest');

K_Xtest = K_xy(X_test,X_test);
D_K_Xtest = real(diag(K_Xtest));

N = 512;
gammas=10.^(-5:.25:4);

bar_E_train = zeros(length(gammas),1);
bar_E_test = zeros(length(gammas),1);

iter_gamma=1;
delta = 0;
for gamma=gammas
    
    delta_tmp=1;
    while abs(delta-delta_tmp)>1e-6
        delta_tmp=delta;
        delta = 1/n*sum(eig_K_X./(N/n*eig_K_X/(1+delta)+gamma));
    end
    
    eig_bar_K = eig_K_X*(N/n/(1+delta));
    eig_bar_Q = 1./(eig_bar_K+gamma);
    
    bar_E_train(iter_gamma) = gamma^2*sum(abs(U_K_y).^2.*eig_bar_Q.^2.*(1/N*sum(eig_bar_K.*eig_bar_Q.^2)/(1-1/N*sum(eig_bar_K.^2.*eig_bar_Q.^2))*eig_bar_K+1))/n;
    bar_E_test(iter_gamma) = 1/n_test*sum( (y_test - (N/n/(1+delta))*U_K_XXtest'*(eig_bar_Q.*U_K_y)).^2 ) + 1/N*sum(abs(U_K_y).^2.*(eig_bar_Q.^2.*eig_bar_K))/(1-1/N*sum(eig_bar_K.^2.*eig_bar_Q.^2))*(1/n_test*(N/n/(1+delta))*sum(D_K_Xtest)-gamma/n_test*sum(eig_bar_Q.^2.*( (N/n/(1+delta))^2*D_U_K_XXtest_2 ))-1/n_test*sum(eig_bar_Q.*( (N/n/(1+delta))^2*D_U_K_XXtest_2 )));
    
    iter_gamma=iter_gamma+1;
end

toc

%% Empirical evaluation over 30 runs
tic
loops=30;

E_train = zeros(length(gammas),loops);
E_test = zeros(length(gammas),loops);

for loop=1:loops
    
    switch W_choice
        case 'gauss'
            W = randn(N,p);
            
        case 'bern'        %%% Bernoulli with pairs (1,.5),(-1,.5)
            W = sign(randn(N,p));
            
        case 'skewed_bern' %%% Bernoulli with pairs (1/sqrt(3),.75),(-sqrt(3),.25)
            Z = rand(N,p);
            W = (Z<3/4)*(1/sqrt(3))+(Z>3/4)*(-sqrt(3));
            
        case 'student' %%% student-t with param nu_student
            W = trnd(nu_student,N,p)/sqrt(nu_student/(nu_student-2));
    end
    
    
    Sigma = sig(W*X);
    Sigma_test = sig(W*X_test);
    
    iter_gamma=1;
    for gamma=gammas
        
        inv_tQ_r = (Sigma'*Sigma/n+gamma*eye(n))\y;
        beta = Sigma/n*inv_tQ_r;
        
        E_train(iter_gamma,loop)=norm(y-Sigma'*beta)^2/n;
        E_test(iter_gamma,loop)=norm(y_test-Sigma_test'*beta)^2/n_test;
        
        iter_gamma=iter_gamma+1;
    end
end
toc

figure;
loglog(gammas,bar_E_train,'b');
hold on;
loglog(gammas,bar_E_test,'r--');
loglog(gammas,mean(E_train,2),'ob');
loglog(gammas,mean(E_test,2),'xr');
legend('$\bar E_{train}$', '$\bar E_{test}$', '$E_{train}$', '$E_{test}$', 'Interpreter', 'latex')

%% Double descent test curve
% Empirical versus theoretical training and test error as a function of N/n
Ns = floor(n*(0:0.05:3.5));
gamma = 1e-7;


bar_E_train = zeros(length(Ns),1);
bar_E_test = zeros(length(Ns),1);

loops=1;
E_train = zeros(length(Ns),loops);
E_test = zeros(length(Ns),loops);

iter_N=1;

for N=Ns
    
    delta = 0;delta_tmp=1; % theoretical
    while abs(delta-delta_tmp)>1e-6
        delta_tmp=delta;
        delta = 1/n*sum(eig_K_X./(N/n*eig_K_X/(1+delta)+gamma));
    end
    
    eig_bar_K = eig_K_X*(N/n/(1+delta));
    eig_bar_Q = 1./(eig_bar_K+gamma);
    
    bar_E_train(iter_N) = gamma^2*sum(abs(U_K_y).^2.*eig_bar_Q.^2.*(1/N*sum(eig_bar_K.*eig_bar_Q.^2)/(1-1/N*sum(eig_bar_K.^2.*eig_bar_Q.^2))*eig_bar_K+1))/n;
    bar_E_test(iter_N) = 1/n_test*sum( (y_test - (N/n/(1+delta))*U_K_XXtest'*(eig_bar_Q.*U_K_y)).^2 ) + 1/N*sum(abs(U_K_y).^2.*(eig_bar_Q.^2.*eig_bar_K))/(1-1/N*sum(eig_bar_K.^2.*eig_bar_Q.^2))*(1/n_test*(N/n/(1+delta))*sum(D_K_Xtest)-gamma/n_test*sum(eig_bar_Q.^2.*( (N/n/(1+delta))^2*D_U_K_XXtest_2 ))-1/n_test*sum(eig_bar_Q.*( (N/n/(1+delta))^2*D_U_K_XXtest_2 )));
    
    for loop=1:loops % empirical
        
        switch W_choice
            case 'gauss'
                W = randn(N,p);
                
            case 'bern'        %%% Bernoulli with pairs (1,.5),(-1,.5)
                W = sign(randn(N,p));
                
            case 'skewed_bern' %%% Bernoulli with pairs (1/sqrt(3),.75),(-sqrt(3),.25)
                Z = rand(N,p);
                W = (Z<3/4)*(1/sqrt(3))+(Z>3/4)*(-sqrt(3));
                
            case 'student' %%% student-t with param nu_student
                W = trnd(nu_student,N,p)/sqrt(nu_student/(nu_student-2));
        end
        
        
        Sigma = sig(W*X);
        Sigma_test = sig(W*X_test);
        
        inv_tQ_r = (Sigma'*Sigma/n+gamma*eye(n))\y;
        beta = Sigma/n*inv_tQ_r;
        
        E_train(iter_N,loop)=norm(y-Sigma'*beta)^2/n;
        E_test(iter_N,loop)=norm(y_test-Sigma_test'*beta)^2/n_test;
        
    end
    iter_N=iter_N+1;
end

figure
hold on
plot(Ns/n,bar_E_train,'-')
plot(Ns/n,bar_E_test,'--r')
plot(Ns/n,mean(E_train,2),'xb')
plot(Ns/n,mean(E_test,2),'ob')
xlabel('$N/n$', 'Interpreter', 'latex')
legend('$\bar E_{train}$', '$\bar E_{test}$', '$E_{train}$', '$E_{test}$', 'Interpreter', 'latex')
axis( [ 0, 3.5, 0, 1 ] )
