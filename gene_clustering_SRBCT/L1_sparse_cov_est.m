function [X,out] = L1_sparse_cov_est(r,epsilon);
% Inputs:
%	r: samples 
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
        [S_est,out] = L1_alm(S1,lamdas(l),epsilon,zeros(d));
        FroErr(k,l) = norm(S_est-S2,'fro');
        %figure(2); semilogy(out.et,out.e);
    end
end
[m mi] = min(sum(FroErr));
lamda = lamdas(mi);
% figure(3);subplot(211);plot(lamdas,sum(FroErr),'-*');set(gca,'xscale','log');

S = cov(r);
[X,out] = L1_alm(S,lamda,epsilon,zeros(d));
