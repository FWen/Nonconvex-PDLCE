clear all; close all;clc;

data = csvread('parkinsons.data.txt',1,1);

pd = data(:,17); % parkinsons status
data(:,17)=[];   % feature vectors
d = size(data,2);

data_parkins = data(find(pd>0),:);
data_healthy = data(find(pd<1),:);

n_parkins = size(data_parkins,1);
n_healthy = size(data_healthy,1);

for time=1:1

    J = randperm(n_parkins);
    train_parkins = data_parkins(J(1:49),:);    % for parkinsons training
    test_parkins  = data_parkins(J(50:end),:);  % for parkinsons testing
    J = randperm(n_healthy);
    train_healthy = data_healthy(J(1:16),:);    % for healthy training
    test_healthy  = data_healthy(J(17:end),:);  % for healthy testing


    %------------training--------------------------------------------------------
    m1 = mean(train_parkins);
    m2 = mean(train_healthy);
    [S1,out] = sparse_cov_est(train_parkins,1);
    [S2,out] = sparse_cov_est(train_healthy,1);
    S1 = (S1 + S1')/2;
    S2 = (S2 + S2')/2;
    
%     S1 = cov(train_parkins);
%     S2 = cov(train_healthy);
    % [min(eig(S1)) min(eig(S2))]

    %------------testing--------------------------------------------------------
    qd1 = ones(1,size(test_parkins,1)); % for parkinsons cases
    for k=1:size(test_parkins,1)
        x = test_parkins(k);
        f1 = -0.5*log(det(S1)) - 0.5*(x-m1)*inv(S1)*(x-m1)' + log(49/65);
        f2 = -0.5*log(det(S2)) - 0.5*(x-m2)*inv(S2)*(x-m2)' + log(16/65);
        if f1<f2  % healthy
            qd1(k) = 2;
        end
    end

    qd2 = ones(1,size(test_healthy,1)); % for healthy cases
    for k=1:size(test_healthy,1)
        x = test_healthy(k);
        f1 = log(1/det(S1))/2 - (x-m1)*inv(S1)*(x-m1)'/2 + log(49/65);
        f2 = log(1/det(S2))/2 - (x-m2)*inv(S2)*(x-m2)'/2 + log(16/65);
        if f1<f2  % healthy
            qd2(k) = 2;
        end
    end

    [length(find(qd1>1)) length(find(qd2<2))]
    (length(find(qd1>1))+length(find(qd2<2)))/130

end
% data_testing  = [ ];



