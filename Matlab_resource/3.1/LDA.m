%% Section 3.1.2: Linear discriminant analysis (LDA)
% This page contains simulations in Section 3.1.2
% Hypotheses testing between two Gaussian models: $\mathcal N(\mu_0, C_0)$ versus $\mathcal N(\mu_1, C_1)$

%% Basic setting
close all; clear; clc

testcase = 'kannada-MNIST'; % Among 'GMM', 'MNIST' and 'fashion-MNIST', 'kannada-MNIST'
testcase_option = 'mixed'; % Among 'mean', 'cov', 'orth' and 'mixed'

coeff = 1;
n = 2048*coeff;
n_test = 128*coeff;
cs = [1/2 1/2];
%cs = [1/4 3/4];
k = length(cs);

switch testcase
    case 'GMM'
        p = 512*coeff; 
        
        switch testcase_option % l = 0 or 1
            case 'means'
                means = @(l) [zeros(l+1,1);1;zeros(p-l-2,1)]*3;
                %means = @(l) sqrt(2)*(-1)^l*[ones(p/2,1); -ones(p/2,1)]/sqrt(p);
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
                %covs = @(l) toeplitz((4*l/10).^(0:(p-1)))*(1+l/sqrt(p)*4);
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
        
        selected_labels=[3 4];   
        
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

%% Empirical evaluations
gamma = 1; % regularization parameter

nb_loop = 20;
T_store = zeros(n_test,nb_loop);
accuracy_store = zeros(nb_loop,1);
for data_loop = 1:nb_loop
    
    switch testcase % generate data
        case {'MNIST', 'fashion-MNIST','kannada-MNIST'}
            X=zeros(p,n);
            X_test=zeros(p,n_test);
            for i=1:k % random data picking
                data = selected_data{i}(:,randperm(size(selected_data{i},2)));
                X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=data(:,1:n*cs(i));
                X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n+1:n+n_test*cs(i));
            end
        case 'GMM'
            W=zeros(p,n);
            W_test=zeros(p,n_test);
            for i=1:k
                W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=sqrt(covs(i-1))*randn(p,cs(i)*n);
                W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=sqrt(covs(i-1))*randn(p,cs(i)*n_test);
            end
            
            X=zeros(p,n);
            X_test=zeros(p,n_test);
            for i=1:k
                X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)+means(i-1)*ones(1,cs(i)*n);
                X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)+means(i-1)*ones(1,cs(i)*n_test);
            end
    end
    
    % run LDA
    X_train0 = X(:,1:n*cs(1));
    X_train1 = X(:,n*cs(1)+1:end);
    
    hat_mu0 = X_train0*ones(n*cs(1),1)/(n*cs(1));
    hat_mu1 = X_train1*ones(n*cs(2),1)/(n*cs(2));
    hat_mu = (hat_mu0 + hat_mu1)/2;
    
    P = @(l) eye(cs(l+1)*n) - ones(cs(l+1)*n, cs(l+1)*n)/(cs(l+1)*n);
    hat_C_gamma = ( X_train0*P(0)*(X_train0') + X_train1*P(1)*(X_train1') )/(n-2) + gamma*eye(p);
    
    % decision function
    T = @(x) (x - hat_mu)'*( hat_C_gamma\(hat_mu0 - hat_mu1) );
    T_store(:,data_loop) = T(X_test); 
    
    accuracy_store(data_loop) = sum(T(X_test(:,1:cs(1)*n_test))>0)/(n_test*cs(1))/2+sum(T(X_test(:,cs(1)*n_test+1:end))<0)/(n_test*cs(2))/2;
    data_loop
end

T_store0 = T_store(1:cs(1)*n_test,:);
T_store1 = T_store(cs(1)*n_test+1:end,:);

disp(['Accuracy:', num2str(mean(accuracy_store))]);

%% Theoretical predictions
eigs_C = @(l) eig(covs(l));

z = - gamma;
tilde_g = ones(2,1);
tilde_g_tmp = zeros(2,1);
g = zeros(2,1);

%watch_dog = 1;
while min(abs(tilde_g-tilde_g_tmp))>1e-6 %%&& watch_dog<50
    tilde_g_tmp = tilde_g;
    
    eigs_C_sum = cs(1)*tilde_g(1)*eigs_C(0) + cs(2)*tilde_g(2)*eigs_C(1);
    
    for a = 1:2
        g(a) = -1/z*sum( eigs_C(a-1)./(1 + eigs_C_sum) )/n;
        tilde_g(a) = -1/z/(1+g(a));
    end
    %watch_dog = watch_dog + 1
end
bar_Q = -1/z*inv( eye(p) + cs(1)*tilde_g(1)*covs(0) + cs(2)*tilde_g(2)*covs(1) );

S = gamma^2*[cs(1)*tilde_g(1)^2*trace( covs(0)*bar_Q*covs(0)*bar_Q )/n, cs(2)*tilde_g(1)^2*trace( covs(0)*bar_Q*covs(1)*bar_Q )/n; cs(1)*tilde_g(2)^2*trace( covs(0)*bar_Q*covs(1)*bar_Q )/n, cs(2)*tilde_g(2)^2*trace( covs(1)*bar_Q*covs(1)*bar_Q )/n];
tmp_S = (eye(2) - S)\S;
R = @(ll,l) cs(ll+1)/cs(l+1)*tmp_S(ll+1,l+1);

bar_QCQ = @(l) bar_Q*covs(l)*bar_Q + R(0,l)*bar_Q*covs(0)*bar_Q + R(1,l)*bar_Q*covs(1)*bar_Q;

delta_mu = means(0) - means(1);
theo_mean = @(l) (-1)^l*delta_mu'*bar_Q*delta_mu/2 - g(1)/2 + g(2)/2;
theo_var = @(l) delta_mu'*bar_QCQ(l)*delta_mu + trace(covs(0)*bar_QCQ(l))/(n*cs(1)) + trace(covs(1)*bar_QCQ(l))/(n*cs(2));

edges = linspace(min([T_store0(:);T_store1(:)])-.5,max([T_store0(:);T_store1(:)])+.5,300);

figure
hold on
histogram(T_store0(:),30,'Normalization','pdf');
histogram(T_store1(:),30,'Normalization','pdf');
plot(edges,normpdf(edges, theo_mean(0), sqrt(theo_var(0))),'--b');
plot(edges,normpdf(edges, theo_mean(1), sqrt(theo_var(1))),'--r');

%%
clc
hat_C0 = X_train0*P(0)*(X_train0')/(n-2) + gamma*eye(p);
hat_C1 = X_train1*P(1)*(X_train1')/(n-2) + gamma*eye(p);

x = X_test(:,1);
(x - hat_mu0)'*inv(hat_C0)*(x-hat_mu0)/sqrt(p)
log(det(hat_C0)/det(hat_C1))
%%

clc

xs1 = linspace(min(T_store0(:))-.5, max(T_store0(:))+.5,50);
step1 = xs1(2) - xs1(1);
xs2 = linspace(min(T_store1(:))-.5, max(T_store1(:))+.5,50);
step2 = xs2(2) - xs2(1);

xs1 = xs1 + step1/2;
xs2 = xs2 + step2/2;

histo0 = histc(T_store0(:),xs1);
histo1 = histc(T_store1(:),xs2);


sprintf('(%f,%f)',[xs1',histo0/step1/nb_loop/(cs(1)*n_test)]')
sprintf('(%f,%f)',[xs2',histo1/step2/nb_loop/(cs(2)*n_test)]')
sprintf('(%f,%f)',[xs1',normpdf(xs1, theo_mean(0), sqrt(theo_var(0)))']')
sprintf('(%f,%f)',[xs2',normpdf(xs2, theo_mean(1), sqrt(theo_var(1)))']')
