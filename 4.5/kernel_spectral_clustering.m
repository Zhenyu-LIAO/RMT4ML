%% Section 4.5.1 Application to kernel spectral clustering
% This page contains simulations in Section 4.5.1.

%% Non-informative eigenvector of $L$
close all; clear; clc

coeff = 4;
p = 512*coeff;
n = 128*coeff;

cs = [1/4, 1/4, 1/2];
k = length(cs); % number of classes

test_case = 'means';

switch test_case
    case 'means'
        means = @(i) [zeros(i-1,1);1;zeros(p-i,1)]*5;
        covs  = @(i) eye(p);
    case 'var'
        means = @(i) zeros(p,1);
        covs  = @(i) eye(p)*(1+(-1)^(i)/sqrt(p)*5);
    case 'orth'
        means = @(i) zeros(p,1);
        covs = @(i) toeplitz((4*(i-1)/10).^(0:(p-1)));
    case 'mixed'
        means = @(i) [-ones(p/2,1);ones(p/2,1)]/sqrt(p);
        covs  = @(i) eye(p)*(1+(i-1)/sqrt(p)*10);
end

rng(1004);
W=zeros(p,n);
for i=1:k
    W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=sqrtm(covs(i))*randn(p,cs(i)*n);
end

X=zeros(p,n);
for i=1:k
    X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)+means(i)*ones(1,cs(i)*n);
end

XX = X'*X;

tau = 2;
f = @(t) 4*(t-tau).^2-(t-tau)+4;
K = f((-2*(XX)+diag(XX)*ones(1,n)+ones(n,1)*diag(XX)')/p);

D = diag(K*ones(n,1));
L = n*diag(1./sqrt(diag(D)))*K*diag(1./sqrt(diag(D)));

[V,eigs_L] = eig(L,'vector');
[~,ind] = sort(eigs_L);
eigs_L = eigs_L(ind);
V = V(:,ind);

figure
histogram(eigs_L(1:n-1), 30, 'Normalization', 'pdf', 'EdgeColor', 'white');
title('Eigenvalues of $L$','Interpreter', 'latex');
annotation('textarrow',[0.79,0.79],[.3,0.13],'String','Eig. 3', 'Interpreter', 'latex')

figure
for i=1:4
    subplot(4,1,i)
    if i ==3
        plot(V(:,n-i+1),'r')
        xlim([1 n])
        xline(n*cs(1),'--')
        xline(n*(cs(1)+cs(2)),'--')
    else
        plot(V(:,n-i+1),'b')
        xlim([1 n])
        xline(n*cs(1),'--')
        xline(n*(cs(1)+cs(2)),'--')
    end
    set(gca,'xtick',[], 'ytick',[])
    
    xlabel(['Eignvector ',num2str(i)], 'Interpreter', 'latex');
end

%% Separation with covariance trace information
close all; clear; clc

coeff = 6;
p = 512*coeff;
n = 128*coeff;

cs = [1/4, 1/4, 1/2];
k = length(cs); % number of classes

test_case = 'var';

switch test_case
    case 'means'
        means = @(i) [zeros(i-1,1);1;zeros(p-i,1)]*5;
        covs  = @(i) eye(p);
    case 'var'
        means = @(i) zeros(p,1);
        covs  = @(i) eye(p)*(1+4*(i-1)/sqrt(p));
    case 'orth'
        means = @(i) zeros(p,1);
        covs = @(i) toeplitz((4*(i-1)/10).^(0:(p-1)));
    case 'mixed'
        means = @(i) [-ones(p/2,1);ones(p/2,1)]/sqrt(p);
        covs  = @(i) eye(p)*(1+(i-1)/sqrt(p)*10);
end

rng(928);
W=zeros(p,n);
for i=1:k
    W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=sqrtm(covs(i))*randn(p,cs(i)*n);
end

X=zeros(p,n);
for i=1:k
    X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)+means(i)*ones(1,cs(i)*n);
end


XX = X'*X;

tau = 0;
for a = 1:k
    tau = tau + 2*cs(a)*trace(covs(a))/p;
end

f = @(t) 1.5*(t-tau).^2-(t-tau)+5;
K = f((-2*(XX)+diag(XX)*ones(1,n)+ones(n,1)*diag(XX)')/p);

D = diag(K*ones(n,1));
L = n*diag(1./sqrt(diag(D)))*K*diag(1./sqrt(diag(D)));

[V,eigs_L] = eig(L,'vector');
[~,ind] = sort(eigs_L);
V = V(:,ind);

v1 = V(:,n);
v2 = V(:,n-1);
v3 = V(:,n-2);

switch test_case
    case 'means'
        figure
        hold on
        plot(v2(1:n*cs(1)),v3(1:n*cs(1)),'rx')
        plot(v2(n*cs(1)+1:n-n*cs(3)),v3(n*cs(1)+1:n-n*cs(3)),'bx')
        plot(v2(n-n*cs(3)+1:n),v3(n-n*cs(3)+1:n),'kx')
        set(gca,'xtick',[], 'ytick',[])
        xlabel('Eignvector $2$', 'Interpreter', 'latex');
        ylabel('Eignvector $3$', 'Interpreter', 'latex');
    case 'var'
        figure
        hold on
        plot(v1(1:n*cs(1)),v2(1:n*cs(1)),'rx')
        plot(v1(n*cs(1)+1:n-n*cs(3)),v2(n*cs(1)+1:n-n*cs(3)),'bx')
        plot(v1(n-n*cs(3)+1:n),v2(n-n*cs(3)+1:n),'kx')
        set(gca,'xtick',[], 'ytick',[])
        xlabel('Eignvector $1$', 'Interpreter', 'latex');
        ylabel('Eignvector $2$', 'Interpreter', 'latex');
end

%% Implementation on MNIST data
clc; close all; clear;

n = 192;

cs = [1/3, 1/3, 1/3];
k = length(cs); % number of classes

init_data = loadMNISTImages('../datasets/MNIST/train-images-idx3-ubyte');
init_labels = loadMNISTLabels('../datasets/MNIST/train-labels-idx1-ubyte');

[labels,idx_init_labels]=sort(init_labels,'ascend');
data=init_data(:,idx_init_labels);

init_n=length(data(1,:));
p=length(data(:,1));

selected_labels=[0 1 2];

if length(selected_labels) ~= k
    error('Error: selected labels and nb of classes not equal!')
end

% Data preprecessing
data = data/max(data(:));
mean_data=mean(data,2);
norm2_data=0;
for i=1:init_n
    norm2_data=norm2_data+1/init_n*norm(data(:,i)-mean_data)^2;
end
data=(data-mean_data*ones(1,size(data,2)))/sqrt(norm2_data)*sqrt(p);


selected_data = cell(k,1);
j=1;
for i=selected_labels
    selected_data{j}=data(:,labels==i);
    j = j+1;
end

X=zeros(p,n);
for i = 1:k
    data = selected_data{i};
    X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=data(:,1:n*cs(i));
end

XX=X'*X;
K=exp(-1/2/p*(-2*XX+diag(XX)*ones(1,n)+ones(n,1)*(diag(XX)')));

D = diag(K*ones(n,1));
L = n*diag(1./sqrt(diag(D)))*K*diag(1./sqrt(diag(D)));

[V,eigs_L] = eig(L,'vector');
[~,ind] = sort(eigs_L);
eigs_L = eigs_L(ind);
V = V(:,ind);

figure
for i=1:4
    subplot(4,1,i)
    plot(V(:,n-i+1),'b')
    set(gca,'xtick',[], 'ytick',[]);
    
    xlabel(['Eignvector ',num2str(i)], 'Interpreter', 'latex');
end

%% $\alpha-\beta$ distance kernel on Gaussian data
close all; clear; clc

coeff = 1;
p = 400*coeff;
n = 1000*coeff;

cs = [1/2 1/2];
k = length(cs); % number of classes

rng(928);
Z = cell(k,1);
for i = 1:k
    Z{i} = randn(p,p/2);
end

means = @(i) zeros(p,1);
covs = @(i) .1*eye(p) + 2*Z{i}*(Z{i})'/p;

covs_mean = cs(1)*covs(1) + cs(2)*covs(2);
%tau = 2*trace(covs_mean)/p;

W=zeros(p,n);
for i=1:k
    W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=sqrtm(covs(i))*randn(p,cs(i)*n);
end

X=zeros(p,n);
for i=1:k
    X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=W(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)+means(i)*ones(1,cs(i)*n);
end

XX = X'*X;

K1= exp(-(-2*XX+diag(XX)*ones(1,n)+ones(n,1)*(diag(XX)'))/p);
K2 = ((-2*XX+diag(XX)*ones(1,n)+ones(n,1)*diag(XX)')/p-2).^2;


[V1,eigs_K1] = eig(K1,'vector');
[V2,eigs_K2] = eig(K2,'vector');
[~,ind] = sort(eigs_K1);
V1 = V1(:,ind);
[~,ind] = sort(eigs_K2);
V2 = V2(:,ind);

v1_1 = V1(:,n);
v1_2 = V1(:,n-1);
v2_1 = V2(:,n);
v2_2 = V2(:,n-1);

figure
subplot(1,2,1)
hold on
plot(v1_1(1:n*cs(1)),v1_2(1:n*cs(1)),'rx')
plot(v1_1(n*cs(1)+1:n),v1_2(n*cs(1)+1:n),'bx')
set(gca,'xtick',[], 'ytick',[])
xlabel('Eignvector $1$', 'Interpreter', 'latex');
ylabel('Eignvector $2$', 'Interpreter', 'latex');

subplot(1,2,2)
hold on
plot(v2_1(1:n*cs(1)),v2_2(1:n*cs(1)),'rx')
plot(v2_1(n*cs(1)+1:n),v2_2(n*cs(1)+1:n),'bx')
set(gca,'xtick',[], 'ytick',[])
xlabel('Eignvector $1$', 'Interpreter', 'latex');
ylabel('Eignvector $2$', 'Interpreter', 'latex');

%% $\alpha-\beta$ distance kernel on EEG data
clc; close all; clear;

n = 1000;

cs = [1/2, 1/2];
k = length(cs); % number of classes

load ../datasets/EEG.mat
init_data = EEG_data;
init_labels = EEG_labels;


[labels,idx_init_labels]=sort(init_labels,'ascend');
data=init_data(:,idx_init_labels);

init_n=length(data(1,:));
p=length(data(:,1));

selected_labels=[2 5];

if length(selected_labels) ~= k
    error('Error: selected labels and nb of classes not equal!')
end

% Data preprecessing
data = data/max(data(:));
mean_data=mean(data,2);
data=data-mean_data*ones(1,size(data,2));

for i=1:init_n
    data(:,i) = data(:,i)/norm(data(:,i))*sqrt(p);
end

selected_data = cell(k,1);
cascade_selected_data=[];
j=1;
for i=selected_labels
    selected_data{j}=data(:,labels==i);
    j = j+1;
end


X=zeros(p,n);
for i = 1:k
    
    data = selected_data{i}(:,randperm(size(selected_data{i},2)));
    X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=data(:,1:n*cs(i));
end

P = eye(n) - ones(n)/n;
X = X*P;
XX = X'*X;

dist_matrix = (-2*XX+diag(XX)*ones(1,n)+ones(n,1)*(diag(XX)'))/p;
tau_estim = sum(dist_matrix(:))/n/(n-1);

K1 = exp(-dist_matrix);
K2 =(dist_matrix-tau_estim).^2;


[V1,eigs_K1] = eig(K1,'vector');
[V2,eigs_K2] = eig(K2,'vector');
[~,ind] = sort(eigs_K1);
V1 = V1(:,ind);
[~,ind] = sort(eigs_K2);
V2 = V2(:,ind);

v1_1 = V1(:,n);
v1_2 = V1(:,n-1);
v2_1 = V2(:,n);
v2_2 = V2(:,n-1);

figure
subplot(1,2,1)
hold on
plot(v1_1(1:n*cs(1)),v1_2(1:n*cs(1)),'rx')
plot(v1_1(n*cs(1)+1:n),v1_2(n*cs(1)+1:n),'bx')
set(gca,'xtick',[], 'ytick',[])
xlabel('Eignvector $1$', 'Interpreter', 'latex');
ylabel('Eignvector $2$', 'Interpreter', 'latex');
title('Top eigenvectors of Gaussian kernel', 'Interpreter', 'latex');

subplot(1,2,2)
hold on
plot(v2_1(1:n*cs(1)),v2_2(1:n*cs(1)),'rx')
plot(v2_1(n*cs(1)+1:n),v2_2(n*cs(1)+1:n),'bx')
set(gca,'xtick',[], 'ytick',[])
xlabel('Eignvector $1$', 'Interpreter', 'latex');
ylabel('Eignvector $2$', 'Interpreter', 'latex');
title('Top eigenvectors of $\alpha-\beta$ kernel', 'Interpreter', 'latex');

%% Performance of spectral clustering with properly scaling kernels
clc; close all; clear;

n = 512;
cs = [1/2 1/2];
ns = n*cs;
k = length(cs); % number of classes

test_case = 'EEG'; % 'MNIST' or 'EEG'

switch test_case
    case 'MNIST'
        init_data = loadMNISTImages('../datasets/MNIST/train-images-idx3-ubyte');
        init_labels = loadMNISTLabels('../datasets/MNIST/train-labels-idx1-ubyte');
        selected_labels=[1 7]; % [3 6]
    case 'EEG'
        load ../datasets/EEG.mat
        init_data = EEG_data;
        init_labels = EEG_labels;
        selected_labels=[1 5];
end

[labels,idx_init_labels]=sort(init_labels,'ascend');
data=init_data(:,idx_init_labels);


init_n=length(data(1,:));
p=length(data(:,1));

if length(selected_labels) ~= k
    error('Error: selected labels and nb of classes not equal!')
end

% Data preprecessing
data = data/max(data(:));
mean_data=mean(data,2);
norm2_data=0;
for i=1:init_n
    norm2_data=norm2_data+1/init_n*norm(data(:,i)-mean_data)^2;
end
data=(data-mean_data*ones(1,size(data,2)))/sqrt(norm2_data)*sqrt(p);

selected_data = cell(k,1);
j=1;
for i=selected_labels
    selected_data{j}=data(:,labels==i);
    j = j+1;
end

a1_over_a2_range = 0:25;
nb_average_loop = 50;
nb_eigs = 2;

P = eye(n) - ones(n,n)/n;

K_gauss_error = zeros(nb_average_loop,1);
K_proper_error = zeros(length(a1_over_a2_range),nb_average_loop);

for loop = 1:nb_average_loop
    
    X=zeros(p,n);
    for i = 1:k
        data = selected_data{i}(:,randperm(size(selected_data{i},2)));
        X(:,sum(cs(1:(i-1)))*n+1:sum(cs(1:i))*n)=data(:,1:n*cs(i));
    end
    X = X*P;
    
    XX=X'*X;
    dist_matrix = (-2*XX+diag(XX)*ones(1,n)+ones(n,1)*(diag(XX)'))/p;
    
    K_gauss = exp(-dist_matrix/2);
    K_gauss_error(loop) = spectral_clustering_perf(cs,K_gauss,nb_eigs);
    
    for range_index = 1:length(a1_over_a2_range)
        
        nu = 2;
        a2 = sqrt( nu/(1+a1_over_a2_range(range_index)^2) );
        a1 = a1_over_a2_range(range_index)*a2;
        
        f = @(t) a2*(t.^2-1)/sqrt(2) + a1*t;
        K_proper = f(XX/sqrt(p))/sqrt(p);
        K_proper = K_proper - diag(diag(K_proper));
        
        K_proper_error(range_index,loop) = spectral_clustering_perf(cs,K_proper,nb_eigs);
    end
end

figure
hold on
plot(a1_over_a2_range, mean(K_proper_error,2));
yline(mean(K_gauss_error),'--');
legend('Properly scaling kernel', 'Gaussian kernel',  'FontSize', 15, 'Interpreter', 'latex');
title('Performance of properly scale versus Gaussian kernel', 'Interpreter', 'latex');
xlabel('$a_1/a_2$', 'Interpreter', 'latex');
ylabel('Misclassification rate', 'Interpreter', 'latex');

%% FUNCTIONS
function perf = spectral_clustering_perf(cs,K,nb_eigs)
n = length(K);
k = length(cs);
ns = floor(n*cs);

[V_K,~] = eigs(K,nb_eigs,'largestreal');
%V_K = real(V_K);

V_means=zeros(k,nb_eigs);
for i=1:k
    V_means(i,:)=mean(V_K(sum(ns(1:(i-1)))+1:sum(ns(1:i)),:));
end
kmeans_output = kmeans(V_K,k,'Start', V_means);

vec=zeros(n,1);
tmp=0;
for perm=perms(1:k)'
    for i=1:k
        vec(sum(ns(1:(i-1)))+1:sum(ns(1:i)))=perm(i)*ones(ns(i),1);
    end
    if kmeans_output'*vec>tmp
        tmp=kmeans_output'*vec;
        best_vec=vec;
    end
end
perf = 1-sum(best_vec==kmeans_output)/n;
end
