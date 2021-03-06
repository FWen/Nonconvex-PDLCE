function [X,out] = L1_alm(S,lamda,epsilon,X0,Xtrue);
% Inputs:
%	S: sample covariance
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

if nargin<4 || length(find(diag(X0)==0))>0
	X = zeros(n);
else
    STD_X0 = diag(X0).^(0.5);
    X = diag(1./STD_X0)*X0*diag(1./STD_X0); %initial correlation matrix
end

if nargin<5
    quiet = 1;
else
    STD_Xtrue = diag(Xtrue).^(0.5);
    Xtrue = diag(1./STD_Xtrue)*Xtrue*diag(1./STD_Xtrue);
    quiet = 0;
end

rho = 1;
max_iter = 200;
ABSTOL   = 1e-7;

W = zeros(n); 
V = zeros(n); 

out.e=[]; out.et=[];
tic;

for iter = 1 : max_iter
    Xm1 = X;
    
    V = X - W/rho;
    for k=1:n-1
        V(1+k:n,k) = shrinkage_Lq(V(1+k:n,k), 1, lamda, rho);
        V(k,1+k:n) = V(1+k:n,k)';
        V(k,k) = 1;
    end
    V(n,n) = 1;
    
    X = 1/(1+rho) * (S + rho*V + W);
     
    [E U] = eig(X);
    eigV  = diag(U);
    eigV(find(eigV<epsilon)) = epsilon;
    X = E*diag(eigV)*E';
   
    X = (real(X)+real(X)')/2;
    
    W = W - rho*(X - V);
      
    if ~quiet
        out.et = [out.et toc];
        out.e  = [out.e norm(X-Xm1,'fro')];
        %out.e  = [out.e norm(X-Xtrue,'fro')/norm(Xtrue,'fro')];
    end
    
    %terminate when both primal and dual residuals are small
    if (rho*norm(X-Xm1,'fro') < n*ABSTOL && norm(X-V,'fro') < n*ABSTOL) 
        X = diag(STD_S)*X*diag(STD_S);
        return;
    end
    
    Xm1 = X;
end
X = diag(STD_S)*X*diag(STD_S);

end

