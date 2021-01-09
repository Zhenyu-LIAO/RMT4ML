%% Section 4.5.3 Kernel ridge regression
% This page contains simulations in Section 4.5.3.

close all; clear; clc

testcase='mixed'; % (among means,var,orth,mixed,MNIST,cifar10)
kernel='gauss';  % (among poly,poly_zero,gauss,linear)

k=2;
switch testcase
    case 'means'
        p_over_n=4;
        p=256;
        cs=[1/2,1/2];
        means = @(i) [zeros(i-1,1);1;zeros(p-i,1)]*5;
        covs  = @(i) eye(p);
        
        switch kernel
            case 'poly'
                derivs=[5 -1 3];
            case 'gauss'
                sigma2=1;
            case 'linear'
                a = 10;
        end
        
    case 'var'
        p_over_n=4;
        p=256;
        
        cs=[3/4 1/4];
        
        means = @(i) zeros(p,1);
        covs  = @(i) eye(p)*(1+(i-1)/sqrt(p)*10);
        
        switch kernel
            case 'poly'
                derivs=[5 -1 3];
                
            case 'poly_zero'
                derivs=[5 0 2];
                
            case 'gauss'
                sigma2=1;
                
            case 'linear'
                a = 10;
        end
        
    case 'orth'
        p_over_n=1/4;
        p=2048;
        n = p/p_over_n;
        n_test = n;
        
        cs=[1/2 1/2];
        
        means = @(i) zeros(p,1);
        %covs  = @(i) diag([zeros(1,sum(prop_classes(1:i-1))*p) ones(1,prop_classes(i)*p) zeros(1,sum(prop_classes(i+1:end))*p)]);
        %covs  = @(i) diag([ones(1,sum(prop_classes(1:i-1))*p) 2*ones(1,prop_classes(i)*p) ones(1,sum(prop_classes(i+1:end))*p)]);
        covs = @(i) toeplitz((4*(i-1)/10).^(0:(p-1)));
        
        
        switch kernel
            case 'poly'
                derivs=[4 -1 3];
            case 'poly_zero'
                derivs=[4 0 2];
            case 'gauss'
                sigma2=1;
            case 'linear'
                a = 10;
        end
        
    case 'mixed'
        %p_over_n=2;
        %p=64;
        p_over_n=4;
        p=512;
        n = p/p_over_n;
        n_test = n;
        
        cs=[1/2 1/2];
        
        means = @(i) [zeros(i-1,1);1;zeros(p-i,1)]*3;
        %covs = @(i) eye(p);
        %covs  = @(i) eye(p)*(1+(i-1)/sqrt(p)*10);
        %covs = @(i) toeplitz((4*(i-1)/10).^(0:(p-1)));
        covs = @(i) toeplitz((4*(i-1)/10).^(0:(p-1)))*(1+(i-1)/sqrt(p)*5);
        
        switch kernel
            case 'poly'
                derivs=[5 -5 3];
            case 'poly_zero'
                derivs=[4 0 2];
                %derivs=[5 -5 0];
            case 'gauss'
                sigma2=1;
        end
        
    case {'MNIST','USPS','cifar10'}
        n = 2000;
        n_test = 256;
        
        switch testcase
            case 'MNIST'
                init_images=loadMNISTImages('train-images-idx3-ubyte');
                init_labels=loadMNISTLabels('train-labels-idx1-ubyte');
            
            case 'USPS'
                load USPS;
                init_images = fea';
                init_labels = gnd;
            
            case 'cifar10'
                load('cifar_total_train.mat');
                init_images = double(total_images');
                init_labels = double(total_labels);
        end
        
        [labels,idx_init_labels]=sort(init_labels,'ascend');
        images=init_images(:,idx_init_labels);
        init_n=length(images(1,:));
        
        p=length(images(:,1));
        
        noise_level_dB=-Inf;
        %noise_level_dB=0;
        noise_level=10^(noise_level_dB/10);
        Noise = rand(p,init_n)*sqrt(12)*sqrt(noise_level*var(images(:)));
        
        %%% Add noise to images
        %images=images+Noise;
        
        mean_images=mean(images,2);
        norm2_images=0;
        for i=1:init_n
            norm2_images=norm2_images+1/init_n*norm(images(:,i)-mean_images)^2;
        end
        images=(images-mean_images*ones(1,size(images,2)))/sqrt(norm2_images)*sqrt(p);
        
        %selected_labels=[3 8 9];
        
        selected_labels=[2 3];
        %selected_labels=[8 9];
        selected_images=[];
        j=1;
        for i=selected_labels
            selected_images=[selected_images images(:,labels==i)];
            MNIST{j}=images(:,labels==i);
            j=j+1;
        end
        
        mean_selected_images=mean(selected_images,2);
        norm2_selected_images=mean(sum(abs(selected_images-mean_selected_images*ones(1,length(selected_images))).^2));
        
        for j=1:length(selected_labels)
            MNIST{j}=(MNIST{j}-mean_selected_images*ones(1,size(MNIST{j},2)))/sqrt(norm2_selected_images)*sqrt(p);
        end
        
        means = @(i) mean(MNIST{i},2);
        covs = @(i) 1/length(MNIST{i})*(MNIST{i}*MNIST{i}')-means(i)*means(i)';
        
        %         selected_labels=[1 7];
        %
        %         j=1;
        %         for i=selected_labels
        %             MNIST{j}=images(:,labels==i);
        %             j=j+1;
        %         end
        %
        %         means = @(i) mean(MNIST{i},2);
        %         covs = @(i) 1/length(MNIST{i})*(MNIST{i}*MNIST{i}')-means(i)*means(i)';
        
        cs=[1/2 1/2];
        
        
        switch kernel
            case 'poly'
                %derivs=[5 -1 3];
                derivs=[5 -1 10];
            case 'poly_zero'
                derivs=[1 0 10];
            case 'gauss'
                sigma2=1;
            case 'linear'
                a = 10;
        end
        
        
    case 'adult'
        
        load('adult.mat');
        
        p = size(data{1},1);
        
        n = 1024;
        n_test = n;
        cs = [1/2 1/2];
        ns = floor(cs*n);
        ns_test = floor(cs*n_test);
        
        X = [data{1} data{2}];
        mean_X=mean(X,2);
        norm2_X=0;
        for i=1:size(X,2)
            norm2_X=norm2_X+1/size(X,2)*norm(X(:,i)-mean_X)^2;
        end
        data{1}=(data{1}-mean_X*ones(1,size(data{1},2)))/sqrt(norm2_X)*sqrt(p);
        data{2}=(data{2}-mean_X*ones(1,size(data{2},2)))/sqrt(norm2_X)*sqrt(p);
        
        for i = 1:2
            data_train{i} = data{i}(:,randperm(ns(i)));
            data_test{i} = data{i}(:,ns(i)+randperm(ns_test(i)));
        end
        
        
        means = @(i) mean(data{i},2);
        covs = @(i) 1/size(data{i},2)*(data{i}*data{i}')-means(i)*means(i)';
        
        cs=[1/2 1/2];
        
        switch kernel
            case 'poly'
                %derivs=[5 -1 3];
                derivs=[5 -1 10];
            case 'poly_zero'
                derivs=[1 0 10];
            case 'gauss'
                sigma2=1;
            case 'linear'
                a = 10;
        end
        
end

covs_o=zeros(p);
for i=1:k
    covs_o=covs_o+covs(i)*cs(i);
end
tau_th  = 2/p*trace(covs_o);

loops = 10;
g_ = zeros(n_test,loops);
g_test = zeros(n_test,loops);
success_rate = zeros(1,loops);

for loop=1:loops
    
    %%% Build data matrix X
    
    switch testcase
        case 'MNIST'
            X=zeros(p,n);
            X_test=zeros(p,n_test);
            for i=1:k
                %X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=MNIST{i}(:,1:n*cs(i));
                %X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=MNIST{i}(:,n+1:n+n_test*cs(i));
                
                %%% Random data picking
                data = MNIST{i}(:,randperm(size(MNIST{i},2)));
                X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=data(:,1:n*cs(i));
                X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n+1:n+n_test*cs(i));
                
                %W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)-means(i)*ones(1,cs(i)*n);
            end
            
            clear init_images;
            clear init_labels;
            
            kurtosis=0;
        case 'adult'
            X=zeros(p,n);
            X_test=zeros(p,n_test);
            for i=1:k
                %X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=MNIST{i}(:,1:n*cs(i));
                %X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=MNIST{i}(:,n+1:n+n_test*cs(i));
                
                %%% Random data picking
                X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=data_train{i};
                X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data_test{i};
                %W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)-means(i)*ones(1,cs(i)*n);
            end
            
        otherwise
            W=zeros(p,n);
            W_test=zeros(p,n_test);
            for i=1:k
                W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=covs(i)^(1/2)*randn(p,cs(i)*n);
                %W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=covs(i)^(1/2)*randn(p,cs(i)*n_test);
                W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=covs(i)^(1/2)*randn(p,cs(i)*n_test);
            end
            
            X=zeros(p,n);
            X_test=zeros(p,n_test);
            for i=1:k
                X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)+means(i)*ones(1,cs(i)*n);
                X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)+means(i)*ones(1,cs(i)*n_test);
            end
            
            kurtosis=0;
    end
    
    XX=X'*X/p;
    
    %%% Create Kernel function
    
    tau_est = 2/n*trace(XX);
    tau = tau_est;
    
    switch kernel
        case {'poly','poly_zero'}
            coeffs=zeros(1,length(derivs));
            for i=1:length(derivs);
                coeffs(i)=derivs(length(derivs)+1-i)/factorial(length(derivs)-i);
            end
            f = @(x) polyval(coeffs,x-tau_est);
            
        case 'gauss'
            sigma2=1;
            f = @(x) exp(-x/(2*sigma2));
            derivs_emp=[f(tau_est) -1/(2*sigma2)*f(tau_est) 1/(4*sigma2^2)*f(tau_est)];
            
        case 'linear'
            f = @(x) a1*x+a2;
            derivs=[f(tau_est) a1 0];
            disp(f(tau_est));
            disp('should be positive')
    end
    
    %%% Build Kernel matrix and vector y
    
    K=f(diag(XX)*ones(1,n)+ones(n,1)*diag(XX)'-2*XX);
    %DK = diag(K*ones(n,1));
     
    %y = [-ones(cs(1)*n,1);ones(cs(2)*n,1)];
    y_test = [-ones(cs(1)*n_test,1);ones(cs(2)*n_test,1)];
    y = [-ones(cs(1)*n,1);ones(cs(2)*n,1)];
    y_ = [-ones(cs(1)*n,1)/cs(1);ones(cs(2)*n,1)/cs(2)];
    
    %%% SVM
    %alpha=quadprog(diag(y)*K*diag(y),-ones(n,1),-eye(n),zeros(n,1),y',0);
    %v = K*diag(y)*alpha;
    %b = (mean(v(ns(1)))+mean(v(ns(2))))/2;
    
    %%% LS-SVM
    gamma = 1;
    S = K+n/gamma*eye(n);
    invS_y = S\y;
    invS_y_ = S\y_;
    invS_1 = S\ones(n,1);
    
    b = sum(invS_y)/sum(invS_1);
    %b_ = sum(invS_y_)/sum(invS_1);
    
    %alpha = invS_y-invS_1*b;
    % without b
    alpha = invS_y;
    
    
    %alpha_ = invS_y_-invS_1*b_;
    %g = @(x) sign(f((x'*x)*ones(n,1)+diag(XX)-2*(X'*x))'*diag(y)*alpha+b);
    
    g = @(Y) alpha'*f(diag(XX)*ones(1,size(Y,2))+ones(n,1)*diag(Y'*Y/p)'-2*(X'*Y/p))+b;
    %%% remove b
    %g = @(Y) alpha'*f(diag(XX)*ones(1,size(Y,2))+ones(n,1)*diag(Y'*Y/p)'-2*(X'*Y/p));
    
    % figure;hist(alpha,25);
    % sum(abs(alpha)>1e-5)/n
    g_(:,loop) = g(X_test)';
    %g_test(:,loop) = g_T(X_test)';
    %success_rate(loop) = 1/n_test*(sum(g_(1:cs(1)*n_test,loop)<cs(2)-cs(1))+sum(g_(cs(1)*n_test+1:end,loop)>cs(2)-cs(1)));
    loop
end

switch kernel
    case 'gauss'
        derivs=[f(tau) -1/(2*sigma2)*f(tau) 1/(4*sigma2^2)*f(tau)];
end

E = [mean(reshape(g_(1:cs(1)*n_test,:),loops*cs(1)*n_test,1)),mean(reshape(g_(cs(1)*n_test+1:end,:),loops*cs(2)*n_test,1))]
Var = [var(reshape(g_(1:cs(1)*n_test,:),loops*cs(1)*n_test,1)),var(reshape(g_(cs(1)*n_test+1:end,:),loops*cs(2)*n_test,1))]
t1 = trace(covs(1)-cs(1)*covs(1)-cs(2)*covs(2))/sqrt(p);
t2 = trace(covs(2)-cs(1)*covs(1)-cs(2)*covs(2))/sqrt(p);
%D = -2*derivs(2)/p*(norm(means(1)-means(2)))^2+derivs(3)/p*(t1-t2)^2+4*derivs(3)/p^2*trace((covs(1)-covs(2))^2);
D = -2*derivs(2)/p*(norm(means(1)-means(2)))^2+derivs(3)/p*(t1-t2)^2+2*derivs(3)/p^2*trace((covs(1)-covs(2))^2);
  
mean_th = (cs(2)-cs(1))*[1;1]+2*cs(1)*cs(2)*gamma*D*[-cs(2);cs(1)];
%mean_th_test = gamma*D*[-cs(2);cs(1)];
%cov_th = 8*gamma^2*cs(1)^2*cs(2)^2*(t2-t1)^2*derivs(3)^2/p^3*[trace(covs(1)^2);trace(covs(2)^2)]...
%    +16*gamma^2*cs(1)^2*cs(2)^2*derivs(2)^2/p^2*[(means(2)-means(1))'*covs(1)*(means(2)-means(1));(means(2)-means(1))'*covs(2)*(means(2)-means(1))]...
%    +16*cs(1)*cs(2)*gamma^2*derivs(2)^2/n/p^2*[cs(2)*trace(covs(1)^2)+cs(1)*trace(covs(1)*covs(2));cs(2)*trace(covs(1)*covs(2))+cs(1)*trace(covs(2)^2)];
V11 = (t2-t1)^2*derivs(3)^2/p^3*trace(covs(1)^2);
V12 = (t2-t1)^2*derivs(3)^2/p^3*trace(covs(2)^2);
V21 = 2*derivs(2)^2/p^2*(means(2)-means(1))'*covs(1)*(means(2)-means(1));
V22 = 2*derivs(2)^2/p^2*(means(2)-means(1))'*covs(2)*(means(2)-means(1));
V31 = 2*derivs(2)^2/n/p^2*(trace(covs(1)^2)/cs(1)+trace(covs(1)*covs(2))/cs(2));
V32 = 2*derivs(2)^2/n/p^2*(trace(covs(1)*covs(2))/cs(1)+trace(covs(2)^2)/cs(2));
cov_th= 8*gamma^2*(cs(1)*cs(2))^2*[V11+V21+V31;V12+V22+V32];


(norm(means(2)-means(1)))^2
(trace(covs(2)-covs(1)))^2/p
trace(((covs(2)-covs(1)))^2)/p

xs1=linspace(min(min(g_(1:cs(1)*n_test,:))),max(max(g_(1:cs(1)*n_test,:))),30);
xs2=linspace(min(min(g_(cs(1)*n_test+1:end,:))),max(max(g_(cs(1)*n_test+1:end,:))),30);

figure;
hold on;
step1=xs1(2)-xs1(1);
step2=xs2(2)-xs2(1);
histo1=histc(reshape(g_(1:cs(1)*n_test,:),loops*cs(1)*n_test,1),xs1);
bar(xs1,histo1/step1/loops/(cs(1)*n_test),'b');

histo2=histc(reshape(g_(cs(1)*n_test+1:end,:),loops*cs(2)*n_test,1),xs2);
bar(xs2,histo2/step2/loops/(cs(2)*n_test),'r');

% plot(xs1,normpdf(xs1,mean_th(1),sqrt(cov_th(1))),'b');
% plot(xs2,normpdf(xs2,mean_th(2),sqrt(cov_th(2))),'r');
plot(xs1,normpdf(xs1,mean_th(1),sqrt(cov_th(1))),'c--','LineWidth',2);
plot(xs2,normpdf(xs2,mean_th(2),sqrt(cov_th(2))),'m--','LineWidth',2);
% set(gca,'linewidth',1,'fontsize',15,'fontname','Times');
% legend({'$$g(\mathbf{x})_{\mathbf{x}\in\mathcal{C}_1}$$ histogram',...
%     '$$g(\mathbf{x})_{\mathbf{x}\in\mathcal{C}_2} $$ histogram',...
%     'Gaussian approximation $$G_1$$',...
%     'Gaussian approximation $$G_2$$'},'interpreter','latex');
hold off

%%
% xs1=linspace(min(min(g_test(1:cs(1)*n_test,:))),max(max(g_test(1:cs(1)*n_test,:))),50);
% xs2=linspace(min(min(g_test(cs(1)*n_test+1:end,:))),max(max(g_test(cs(1)*n_test+1:end,:))),50);
% figure;
% hold on;
% step1=xs1(2)-xs1(1);
% step2=xs2(2)-xs2(1);
% histo1=histc(reshape(g_test(1:cs(1)*n_test,:),loops*cs(1)*n_test,1),xs1);
% bar(xs1,histo1/step1/loops/(cs(1)*n_test),'b');
% 
% histo2=histc(reshape(g_test(cs(1)*n_test+1:end,:),loops*cs(2)*n_test,1),xs2);
% bar(xs2,histo2/step2/loops/(cs(2)*n_test),'r');
% 
% 
% plot(xs1,normpdf(xs1,mean_th_test(1),sqrt(cov_th_test(1))),'c--','LineWidth',2);
% plot(xs2,normpdf(xs2,mean_th_test(2),sqrt(cov_th_test(2))),'m--','LineWidth',2);
% set(gca,'linewidth',1,'fontsize',15,'fontname','Times');
% legend({'$$g^{*}(\mathbf{x})_{\mathbf{x}\in\mathcal{C}_1}$$ histogram',...
%     '$$g^{*}(\mathbf{x})_{\mathbf{x}\in\mathcal{C}_2} $$ histogram',...
%     'Gaussian approximation $$G^*_1$$',...
%     'Gaussian approximation $$G^*_2$$'},'interpreter','latex');
% 
% E = [mean(reshape(g_(1:cs(1)*n_test,:),loops*cs(1)*n_test,1)),mean(reshape(g_(cs(1)*n_test+1:end,:),loops*cs(2)*n_test,1))]
% Var = [var(reshape(g_(1:cs(1)*n_test,:),loops*cs(1)*n_test,1)),var(reshape(g_(cs(1)*n_test+1:end,:),loops*cs(2)*n_test,1))]
% hold off
center =  gamma*derivs(1)/(1+gamma*derivs(1))*(cs(2)-cs(1))
disp(eigs(K))

