clc; close all; clear;

testcase = 'MNIST';
testcase_option = 'mixed';


n = 500; % nb of training data (in total)
n_test = 512; % nb of test data (in total)

%cs = [1/3, 1/3, 1/3];
cs = [1/2, 1/2];
n_cs = n*cs;
k = length(cs); % nb of classes

%%% GMM data settings
switch testcase
    case 'GMM'
        p = 250; % dimension of GMM data
        
        switch testcase_option
            case 'means'
                means = @(i) [zeros(i-1,1);1;zeros(p-i,1)]*5;
                covs  = @(i) eye(p);
            case 'var'
                means = @(i) zeros(p,1);
                covs  = @(i) eye(p)*(1+(i-1)/sqrt(p)*10);
            case 'orth'
                means = @(i) zeros(p,1);
                covs = @(i) toeplitz((4*(i-1)/10).^(0:(p-1)));
            case 'mixed'
                means = @(i) (-1)^(i)*[1;zeros(p-1,1)]*2;
                covs = @(i) eye(p);
                %means = @(i) [zeros(i-1,1);1;zeros(p-i,1)]*2;
                %covs  = @(i) eye(p)*(1+(i-1)/sqrt(p)*5);
                %covs = @(i) toeplitz((4*(i-1)/10).^(0:(p-1)))*(1+(i-1)/sqrt(p)*4);
        end
        
    case 'EEG'
        load datasets/EEG_data.mat
        init_data = EEG_data;
        init_labels = EEG_labels;
    case 'MNIST'
%         init_data = loadMNISTImages('./datasets/MNIST/t10k-images-idx3-ubyte');
%         init_labels = loadMNISTLabels('./datasets/MNIST/t10k-labels-idx1-ubyte');
        init_data = loadMNISTImages('./datasets/MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('./datasets/MNIST/train-labels-idx1-ubyte');
    case 'fashion-MNIST'
        init_data = loadMNISTImages('./datasets/fashion-MNIST/t10k-images-idx3-ubyte');
        init_labels = loadMNISTLabels('./datasets/fashion-MNIST/t10k-labels-idx1-ubyte');
    case 'SVHN'
        load datasets/SVHN_data.mat
        init_data = data;
        init_labels = y;
    case 'CIFAR'
        load datasets/CIFAR/cifar-10-batches-mat/data_batch_1.mat
        init_data = double(data');
        init_labels = double(labels);
end

switch testcase
    case {'EEG', 'MNIST', 'fashion-MNIST', 'SVHN', 'CIFAR'}
        [labels,idx_init_labels]=sort(init_labels,'ascend');
        data=init_data(:,idx_init_labels);
        
        init_n=length(data(1,:));
        p=length(data(:,1));
        
        %selected_labels=[0 6]; %%% fashion-MNIST
        %selected_labels=[5 7]; %%% fashion-MNIST kernel clustering
        %selected_labels=[7 9]; %%% MNIST
        selected_labels=[8 9]; %%% MNIST kernel clustering
        
        if length(selected_labels) ~= k
            error('Error: selected labels and nb of classes not equal!')
        end
        
        %%% Add Gaussian noise to data
        %noise_level_dB=0;
        %noise_level=10^(noise_level_dB/10);
        %Noise = rand(p,init_n)*sqrt(12)*sqrt(noise_level*var(data(:)));
        %images=images+Noise;
        
        %%% Data preprecessing
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
        
        % for large dimensional data, consider to store the means and
        % covariance in the memory to speed up
        means = @(i) mean(selected_data{i},2);
        covs = @(i) 1/length(selected_data{i})*(selected_data{i}*selected_data{i}')-means(i)*means(i)';
end

%%% build data matrix X

%%% when you want to average over a lot of data
% nb_data_loop = 50;
% success_rate = zeros(1,nb_data_loop);
% for data_loop = 1:nb_data_loop

switch testcase
    case {'EEG', 'MNIST', 'fashion-MNIST', 'SVHN', 'CIFAR'}
        X=zeros(p,n);
        X_test=zeros(p,n_test);
        for i=1:k
            %%% Random data picking
            data = selected_data{i}(:,randperm(size(selected_data{i},2)));
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=data(:,1:n*cs(i));
            %X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=data(:,n+1:n+n_test*cs(i));
            %X(:,sum(n_cs(1:(i-1)))+1:sum(n_cs(1:i))) =data(:,1:n_cs(i));
            %X_test(:,sum(n_cs(1:(i-1)))+1:sum(n_cs(1:i))) =data(:,1:n_cs(i));
            
            %W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)-means(i)*ones(1,cs(i)*n);
            %W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)-means(i)*ones(1,cs(i)*n_test);
        end
        
    case 'GMM'
        W=zeros(p,n);
        W_test=zeros(p,n_test);
        for i=1:k
            [U,S] = svd(covs(i));
            W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=U*S^(1/2)*(U')*randn(p,cs(i)*n);
            W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=U*S^(1/2)*(U')*randn(p,cs(i)*n_test);
        end
        
        X=zeros(p,n);
        X_test=zeros(p,n_test);
        for i=1:k
            X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)+means(i)*ones(1,cs(i)*n);
            X_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)=W_test(:,sum(cs(1:(i-1)))*n_test+1:sum(cs(1:i))*n_test)+means(i)*ones(1,cs(i)*n_test);
        end
end

%%% your algorithm and theory starts here

XX=X'*X;
d=diag(XX);
K=exp(-1/2/p*(-2*XX+d*ones(1,n)+ones(n,1)*d'));
%K=exp(-1/2/p*(XX-diag(d)));
% K = (XX/sqrt(p))/sqrt(p);
% K = K - diag(diag(K));

% K = zeros(n,n);
% for i=1:n
%     for j=1:n
%         K(i,j) = dot(X(:,i),X(:,j))/norm(X(:,i))/norm(X(:,j));
%     end
% end

eig_K = eig(K);
eig_K = eig_K(2:end);

xs = linspace(.2,max(eig_K)*1.1,30);
step = xs(2) - xs(1);
histo = histc(eig_K, xs);

% figure
% bar(xs, histo/step/n);
% title('eigenvalues of K')
% sprintf('(%f,%f)',[xs',histo/step/n]')

% %% (histogram) concentration of distance and angle between data
% distance_matrix = (-2*XX+d*ones(1,n)+ones(n,1)*d')/p;
% distance_same1 = distance_matrix(1:n/2,1:n/2);
% distance_same2 = distance_matrix(n/2+1:end,n/2+1:end);
% distance_diff1 = distance_matrix(1:n/2,n/2+1:end);
% distance_diff2 = distance_matrix(n/2+1:end,1:n/2);
% 
% distance_same = [distance_same1(:);distance_same2(:)];
% distance_diff = [distance_diff1(:);distance_diff2(:)];
% 
% figure
% hold on
% histogram(distance_same, 'FaceColor','r');
% histogram(distance_diff, 'FaceColor', 'b');

%%
figure
colormap gray
%imagesc(K)
imshow(K,'border','tight','initialmagnification','fit');
axis normal
axis off
%saveas(gcf,'kernel4','png')

% fileID = fopen('kernel1.txt','w');
% for i = 1:n
%     fprintf(fileID, '%.2f %.2f %.2f \n', [(0:n-1)',(n-1-i)*ones(n,1), K(i,:)']');
%     fprintf(fileID, '\n');
% end
% fclose(fileID);
clc
[U,D]=eigs(K,3);
step = 1;
index = 3;
figure
plot(U(1:step:end,index))
output = U(1:step:end, index)';

sprintf('%d %f \n',[(1:length(output))', output']')
%%

[U L]=eigs(K,2);
figure;
hold on;
plot(U(1:n/2,1),U(1:n/2,2),'bo');
plot(U(n/2+1:n,1),U(n/2+1:n,2),'rx');

[U2 L2]=eigs(K2,2);
figure;
hold on;
plot(U2(1:n/2,1),U2(1:n/2,2),'bo');
plot(U2(n/2+1:n,1),U2(n/2+1:n,2),'rx');

%% visualization clusters in small and large dimensional problems
close all; clear; clc
%%% small dim
p = 250;
n = 500;
means = @(a) [zeros(a-1,1);1;zeros(p-a,1)]*5;

J = [ones(n/2,1), zeros(n/2,1); zeros(n/2,1), ones(n/2,1)];
M = [means(1), means(2)];
Z = randn(p,n);
X = Z + M*(J');

%XX = X'*X/n;
V = pca(X, 'Algorithm', 'eig');

figure
plot(V(:,1),V(:,2),'*')
%% concentrated random vectors visualization
n=500;
r = 1+randn(n,1)/sqrt(n);
theta = rand(n,1)*2*pi;

x = r.*cos(theta)-2;
y = r.*sin(theta);

figure
plot(x,y,'x')
sprintf('(%.3f, %.3f)',[x,y]')
