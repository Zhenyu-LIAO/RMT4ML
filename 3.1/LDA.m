%% Section 3.1.2: Linear discriminant analysis (LDA)
% This page contains simulations in Section 3.1.2.

%% Basic settings
close all; clear; clc

testcase = 'Fashion-MNIST'; % Among 'GMM', 'MNIST', 'Fashion-MNIST', 'Kannada-MNIST', 'Kuzushiji-MNIST'
switch testcase
    case 'GMM'
        testcase_option = 'mixed'; % when testcase = 'GMM', among 'mean', 'cov', 'orth' and 'mixed'
end

coeff = 1;
n = 2048*coeff;
n_test = 128*coeff;
cs = [1/2 1/2]; 
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
                means = @(l) [zeros(l+1,1);1;zeros(p-l-2,1)]*3;
                covs = @(l) toeplitz((4*l/10).^(0:(p-1)));
                %covs  = @(l) eye(p)*(1+l/sqrt(p)*20);
                %covs = @(l) toeplitz((4*l/10).^(0:(p-1)))*(1+l/sqrt(p)*4);
        end
        
    case 'MNIST'
        init_data = loadMNISTImages('../datasets/MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('../datasets//MNIST/train-labels-idx1-ubyte');
    case 'Fashion-MNIST'
        init_data = loadMNISTImages('../datasets/Fashion-MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('../datasets/Fashion-MNIST/train-labels-idx1-ubyte');
    case 'Kannada-MNIST'
        init_data = loadMNISTImages('../datasets/Kannada-MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('../datasets/Kannada-MNIST/train-labels-idx1-ubyte');
    case 'Kuzushiji-MNIST'
        init_data = loadMNISTImages('../datasets/Kuzushiji-MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('../datasets/Kuzushiji-MNIST/train-labels-idx1-ubyte');
end

switch testcase % real-world data pre-processing 
    case {'MNIST', 'Fashion-MNIST','Kannada-MNIST','Kuzushiji-MNIST'}
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

%% Empirical evaluation of LDA
gamma = .1; % regularization parameter

nb_loop = 30;
T_store = zeros(n_test,nb_loop);
accuracy_store = zeros(nb_loop,1);
for data_loop = 1:nb_loop
    
    switch testcase % generate data
        case {'MNIST', 'Fashion-MNIST','Kannada-MNIST','Kuzushiji-MNIST'}
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
                W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=sqrtm(covs(i-1))*randn(p,cs(i)*n);
                W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=sqrtm(covs(i-1))*randn(p,cs(i)*n_test);
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
end

T_store0 = T_store(1:cs(1)*n_test,:);
T_store1 = T_store(cs(1)*n_test+1:end,:);

disp(['Classif accuracy:', num2str(mean(accuracy_store))]);

%% Theoretical prediction of LDA decision (soft) output
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
    
end
bar_Q = -1/z*inv( eye(p) + cs(1)*tilde_g(1)*covs(0) + cs(2)*tilde_g(2)*covs(1) );

S = gamma^2*[cs(1)*tilde_g(1)^2*trace( covs(0)*bar_Q*covs(0)*bar_Q )/n, cs(2)*tilde_g(1)^2*trace( covs(0)*bar_Q*covs(1)*bar_Q )/n; cs(1)*tilde_g(2)^2*trace( covs(0)*bar_Q*covs(1)*bar_Q )/n, cs(2)*tilde_g(2)^2*trace( covs(1)*bar_Q*covs(1)*bar_Q )/n];
tmp_S = (eye(2) - S)\S;
R = @(ll,l) cs(ll+1)/cs(l+1)*tmp_S(ll+1,l+1);

bar_QCQ = @(l) bar_Q*covs(l)*bar_Q + R(0,l)*bar_Q*covs(0)*bar_Q + R(1,l)*bar_Q*covs(1)*bar_Q;

delta_mu = means(0) - means(1);
theo_mean = @(l) (-1)^l*delta_mu'*bar_Q*delta_mu/2 - g(1)/2/cs(1) + g(2)/2/cs(2);
theo_var = @(l) delta_mu'*bar_QCQ(l)*delta_mu + trace(covs(0)*bar_QCQ(l))/(n*cs(1)) + trace(covs(1)*bar_QCQ(l))/(n*cs(2));

edges = linspace(min([T_store0(:);T_store1(:)])-.5,max([T_store0(:);T_store1(:)])+.5,300);

figure
hold on
histogram(T_store0(:),30,'Normalization','pdf','EdgeColor', 'white');
histogram(T_store1(:),30,'Normalization','pdf','EdgeColor', 'white');
plot(edges,normpdf(edges, theo_mean(0), sqrt(theo_var(0))),'--b');
plot(edges,normpdf(edges, theo_mean(1), sqrt(theo_var(1))),'--r');
legend('Empirical $T(x\sim \mathcal H_0)$', 'Empirical $T(x\sim \mathcal H_1)$', 'Theory $T(x\sim \mathcal H_0)$', 'Theory $T(x\sim \mathcal H_1)$', 'Interpreter','latex', 'FontSize', 15)


%% FUNCTION
function images = loadMNISTImages(filename)
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images
%from 

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end

function labels = loadMNISTLabels(filename)
%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end
