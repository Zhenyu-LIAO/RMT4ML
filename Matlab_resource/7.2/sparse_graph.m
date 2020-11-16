%% Section 7.2: From dense to sparse graphs: a different approach.
% This page contains simulations in Section 7.2.

%% Complex spectrum of the non-backtracking matrix $N$
close all; clear; clc

n = 1000;

p_in = 12;
p_out = 1;
cs = [1/2 1/2]';
k = length(cs);

A11 = binornd(1,p_in/n,n*cs(1),n*cs(1));
A11 = tril(A11,-1)+tril(A11,-1)';
A22 = binornd(1,p_in/n,n*cs(2),n*cs(2));
A22 = tril(A22,-1)+tril(A22,-1)';
A12 = binornd(1,p_out/n,n*cs(1),n*cs(2));

A = [A11 A12; A12' A22];
A = A - diag(A);

% get the (directed) edges from A
[I,J] = ind2sub(size(A),find(triu(A,1)>0));
E = [I,J;J,I];
m = length(E);
N = zeros(m,m);

for i=1:m
    for j=1:m
        if E(i,2)==E(j,1) && E(i,1)~=E(j,2)
            N(i,j) = 1;
        end
    end
end

eigs_N = eig(N);

figure
hold on
plot(eigs_N,'x')
