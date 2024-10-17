clear; rng('default');
addpath(genpath(pwd));

Z0=csvread('Z0.csv',1,0); %Z0 is the response matrix of n*p
iU=csvread('iU.csv',1,0); %iU is the rescaled covariate matrix of n*q
[n,p]=size(Z0);
q=size(iU,2)-1;
dim=(p-1)*(q+1);
A=sparse(n*p,p*dim); %this is the large design matrix, which is sparse
b=[];
interM = [];
for j=1:(q+1)
    interM=[interM Z0.*iU(:,j)]; %these are the interaction terms between Z0 and iU
end

%% populates the design matrix with nonzero blocks
for j=1:p 
	Xj=interM;
    Xj(:,((0:q)*p+j)) = [];
    A(((j-1)*n+1):(j*n),((j-1)*dim+1):(j*dim)) = Xj;
	y1=Z0(:,j);
	b=[b;y1-mean(y1)];
end

%% group structure
int=repmat(ceil((1:dim)/(p-1)),1,p);
G = [];
for i = 1:(q+1)
    G = [G;find(int==i)'];
end
ind = zeros(3,q+1);
grpsize = round(size(A,2)/(q+1)); 
for i = 1:(q+1)
    if i == 1
        ind(1,1) = 1; ind(2,1) = grpsize; ind(3,1) = 0;
    else
        ind(1,i) = ind(2,i-1) + 1;
        ind(2,i) = i*grpsize;
        ind(3,i) = 1;
    end
end

%% 5-fold cross validation
lam1_max = norm(A'*b,Inf); alpha = 0.75; nl=50;
lambda1  = lam1_max*exp(linspace(log(1), log(0.2), nl)); %try 50 lambda1
lambda2 = lambda1*(1-alpha)/alpha; %we set alpha and do not tune lambda2
Amap = @(x) mexMatvec(A,x,0);
ATmap = @(y) mexMatvec(A,y,1);
AATmap = @(x) Amap(ATmap(x));

eigsopt.issym = 1;
eigsopt.tol=1e-3;
Lip = eigs(AATmap,length(b),1,'LA',eigsopt);

cut = repelem(1:5,n/5); 
Cerror = zeros(5,nl); 
for t=1:5
    foldidx1 = repmat(cut~=t,1,p);
    Atrain = A(foldidx1,:);
    btrain = b(foldidx1);
    foldidx2 = repmat(cut==t,1,p);
    Atest = A(foldidx2,:);
    btest = b(foldidx2);
    Amap = @(x) mexMatvec(Atrain,x,0);
    ATmap = @(y) mexMatvec(Atrain,y,1);
    AATmap = @(x) Amap(ATmap(x));
    opts.stoptol = 1e-4;
    opts.printyes = 0;
    opts.Lip = Lip;
    Ainput.A = Atrain;
    Ainput.Amap = @(x) Amap(x);
    Ainput.ATmap = @(x) ATmap(x);
    %% solver
    parfor i=1:nl
        c = [lambda1(i);lambda2(i)];
        [obj,y,z,x,info,runhist] = SGLasso_SSNAL(Ainput,btrain,size(Atrain,2),c,G,ind,opts);
        supp = find(x~=0);
        A_supp = Atrain(:,supp);
        x_supp = inv(A_supp'*A_supp)*A_supp'*btrain; %we do a refitting here after selection to reduce bias from lasso
        x(supp) = x_supp;
        res = Atest*x - btest;
        Cerror(t,i) = Cerror(t,i)+sum(res.^2);  
    end
end
meanC = mean(Cerror,1); 
[minv,minid]=min(meanC);

%% estimation
c = [lambda1(minid);lambda2(minid)];
Amap = @(x) mexMatvec(A,x,0);
ATmap = @(y) mexMatvec(A,y,1);
AATmap = @(x) Amap(ATmap(x));
opts.Lip = Lip;
opts.stoptol = 1e-6;
Ainput.A = A;
Ainput.Amap = @(x) Amap(x);
Ainput.ATmap = @(x) ATmap(x);
[obj,y,z,x,info,runhist] = SGLasso_SSNAL(Ainput,b,size(A,2),c,G,ind,opts);
supp = find(x~=0);
A_supp = A(:,supp);
x_supp = inv(A_supp'*A_supp)*A_supp'*b; %we do a refitting here after selection to reduce bias from lasso
x(supp) = x_supp;
csvwrite('beta.csv',x);




