function [X,out] = Lq_sparse_cov_est(r,q,epsilon);
% Inputs:
%	r: samples 
%	q: 0<=q<1
% Outputs
%	X: the estimation
%	out.e: the error with respect to the true
%	out.et: time index

d = size(r,2);
n = size(r,1);
n2 = round(n/log(n));
n1 = n - n2;

% cross-validation
lamdas = logspace(log10(1e-4), log10(1), 30);
for k=1:5
    J  = randperm(n);     % m randomly chosen indices
    S1 = cov(r(J(1:n1),:));
    S2 = cov(r(J(n1+1:end),:));
    for l=1:length(lamdas)
        if q<1
            X0 = L1_alm(S1,lamdas(l),epsilon,zeros(d));
        else
            X0 = zeros(d);
        end
        [S_est,out]  = Lq_adm(S1,q,lamdas(l),epsilon,X0);

        FroErr(k,l) = norm(S_est-S2,'fro');
%         figure(2); semilogy(out.et,out.e);
    end
end
[m mi] = min(sum(FroErr));
lamda = lamdas(mi);
% figure(3);plot(lamdas,sum(FroErr),'-*');set(gca,'xscale','log');


S = cov(r);
if q<1
    X0 = L1_alm(S,lamda,epsilon,zeros(d));
else
    X0 = zeros(d);
end
[X,out] = Lq_adm(S,q,lamda,epsilon,X0);
