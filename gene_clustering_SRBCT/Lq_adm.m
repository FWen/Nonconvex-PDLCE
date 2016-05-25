function [X,out] = Lq_adm_(S,q,lamda,epsilon,X0,Xtrue);
% Inputs:
%	S: sample covariance
%	q: 0<=q<=1
%   epsilon: the lower bound for the minimal eigenvalue
%	lamda: regularization parameter 
%   epsilon: the lower bound for the minimal eigenvalue
%	Xtrue: for debug, for calculation of errors
%	X0: initialization 
% Outputs
%	X: the estimation
%	out.e: the error with respect to the true
%	out.et: time index

n = size(S,1);

STD_S = diag(S).^(0.5);
S  = diag(1./STD_S)*S*diag(1./STD_S);   %correlation matrix

if nargin<5 || length(find(diag(X0)==0))>0
	X = zeros(n);
else
    STD_X0 = diag(X0).^(0.5);
    X = diag(1./STD_X0)*X0*diag(1./STD_X0); %initial correlation matrix
end

if nargin<6
    quiet = 1;
else
    STD_Xtrue = diag(Xtrue).^(0.5);
    Xtrue = diag(1./STD_Xtrue)*Xtrue*diag(1./STD_Xtrue);%ground-trueth correlation matrix
    quiet = 0;
end

rho = 1;

max_iter = 400;
ABSTOL   = 1e-7;

V1 = zeros(n); 
V2 = zeros(n); 
ck = 1e-1;
dk = 1e-1;

out.e=[]; out.et=[];
tic;
    
for iter = 1 : max_iter
    Xm1 = X;
    rho = rho*1.08;
    
    % V1 subproblem
    Z1 = (rho*X+ck*V1)/(rho+ck);
    for k=1:n-1
        V1(1+k:n,k) = shrinkage_Lq(Z1(1+k:n,k), q, lamda, rho+ck);
        V1(k,1+k:n) = V1(1+k:n,k)';
        V1(k,k) = 1;
    end
    V1(n,n) = 1;
    
    % V2 subproblem
    Z2 = (rho*X+dk*V2)/(rho+dk);
    [E U] = eig(Z2);
    eigV2 = diag(U);
    eigV2(find(eigV2<epsilon)) = epsilon;
    V2 = E*diag(eigV2)*E';
    
    % X subproblem
    X = 1/(1+2*rho) * (S + rho*(V1 + V2));

    X = (real(X)+real(X)')/2;
    
    if ~quiet
        out.et = [out.et toc];
        out.e  = [out.e norm(X-Xm1,'fro')];
        %out.e  = [out.e norm(X-Xtrue,'fro')/norm(Xtrue,'fro')];
    end
    
    %terminate when both primal and dual residuals are small
    if (norm(X-Xm1,'fro') < n*ABSTOL) 
        X = diag(STD_S)*X*diag(STD_S);
        return;
    end
    
end
X = diag(STD_S)*X*diag(STD_S);

end

