clear all;
n=2000;
p=.7;
qs=.2:.001:p;

loops=2;

dominant_eig = zeros(length(qs),loops);

classif = zeros(length(qs),loops);

i=1;
for q=qs
    for loop=1:loops
        A11=binornd(1,p,n/2,n/2);
        A11=tril(A11,-1)+tril(A11,-1)';
        A22=binornd(1,p,n/2,n/2);
        A22=tril(A22,-1)+tril(A22,-1)';
        A12=binornd(1,q,n/2,n/2);
        
        A=[A11 A12;A12' A22];
        d=A*ones(n,1);
        
        B=1/sqrt(q*(1-q)*n)*(A-d*d'/sum(d));
        
        [u,l]=eigs(B,1);
        dominant_eig(i,loop)=l;
        classif(i,loop)=max(sum(u(1:n/2)>0)+sum(u(n/2+1:n)<0),sum(u(1:n/2)<0)+sum(u(n/2+1:n)>0))/n;
    end
    i=i+1;
end

%%%

%%% theory

M0 = @(q) (eye(2)-1/2*ones(2))*sqrt(n)*(p-q)/q*(eye(2)-1/2*ones(2));
eigM0 = @(q) eigs(M0(q),1);

qs2 = .2:.001:(p-2/sqrt(n))/(1-2/sqrt(n));
ell = @(q) sqrt(n)*(p-q)/2./sqrt(q.*(1-q));
lambda_ell = @(q) (q.*ell(q)+(1-q)./ell(q))./sqrt(q.*(1-q));

figure
hold on;
plot(qs,ell(qs));
eigM0_qs = zeros(1,length(qs));
i=1;
for q=qs
    eigM0_qs(i)=eigM0(q);
    i=i+1;
end
plot(qs,1/2*eigM0_qs,'r');

figure
hold on;
plot(qs,dominant_eig);
plot(qs2,lambda_ell(qs2),'r');

m_sc = @(z) 1/2*(-z+sqrt(z.^2-4));

classif_theo = 1-qfunc(sqrt(max(n*(p-qs).^2/4./(1-qs).^2-1,0)));
classif_theo2 = 1-qfunc(sqrt(max((1-real(m_sc(dominant_eig(:,1))).^2)./real(m_sc(dominant_eig(:,1))).^2,0)));
classif_theo3 = 1-qfunc(sqrt(max((1-real(m_sc(lambda_ell(qs))).^2)./real(m_sc(lambda_ell(qs))).^2,0)));
figure;
hold on;
plot(qs,mean(classif,2));
plot(qs,classif_theo,'r');
plot(qs,classif_theo2,'g');
plot(qs,classif_theo3,'k');