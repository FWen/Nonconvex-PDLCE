clear all; close all;clc;

d  = 100; % covariance dimension

SIGMA = cov_model(d, 1);

figure(1);subplot(3,3,1); imagesc(SIGMA);title('Original');
% [E U]=eig(SIGMA); min(diag(U))

Ns = [100 200 400 800];

for iN=1:length(Ns)

    for time=1:20
        [Ns(iN) time]
        
        r = mvnrnd(zeros(d,1), SIGMA, Ns(iN));
        S = r'*r/Ns(iN);%S = cov(R);
        
        figure(1);subplot(3,3,2); imagesc(S);
        title(['Sample Cov' ', FroErr=' num2str(norm(S-SIGMA,'fro'),'% 10.2f') ', SpeErr=' num2str(norm(S-SIGMA),'% 10.2f')]);   

        
        epsilon = 1e-2;
        
        %--ALM algorithm with L1-norm penalty (soft-thresholding)---------------
        [X1,out] = L1_sparse_cov_est(r,epsilon,SIGMA);
        ErrFro(time,1) = norm(X1-SIGMA,'fro');
        ErrSpe(time,1) = norm(X1-SIGMA);

        figure(1);subplot(3,3,3); imagesc(real(X1));
        title(['L1-ALM' ', FroErr=' num2str(ErrFro(time,1),'% 10.2f') ', SpeErr=' num2str(ErrSpe(time,1),'% 10.2f')]);
        
        %--Proposed ADM algorithm with different penalties---------------
        qs = [1 0.5 0]; % for soft-, L_(0.5)-, and hard-thresholding penalties
        for iq=1:length(qs)
            [X,out] = Lq_sparse_cov_est(r,qs(iq),epsilon,SIGMA);
            ErrFro(time,iq+1) = norm(X-SIGMA,'fro');
            ErrSpe(time,iq+1) = norm(X-SIGMA);
            
            figure(1);subplot(3,3,iq+3); imagesc(X);
            title(['q=' num2str(qs(iq),'% 10.1f') ', FroErr=' num2str(ErrFro(time,iq),'% 10.1f') ', SpeErr=' num2str(ErrSpe(time,iq),'% 10.1f')]);
        end
        
        % for SCAD
        [X,out] = SCAD_sparse_cov_est(r,epsilon,SIGMA);
        ErrFro(time,5) = norm(X-SIGMA,'fro');
        ErrSpe(time,5) = norm(X-SIGMA);
            
        figure(1);subplot(3,3,7); imagesc(X);
        title(['SCAD ', ', FroErr=' num2str(ErrFro(time,iq),'% 10.1f') ', SpeErr=' num2str(ErrSpe(time,iq),'% 10.1f')]);

    end

    AverErrFro(iN,:) = mean(ErrFro);
    AverErrSpe(iN,:) = mean(ErrSpe);
end

figure(4);subplot(1,2,1);
plot(Ns,AverErrFro(:,1),'-',Ns,AverErrFro(:,2),'--+',Ns,AverErrFro(:,3),'-.',Ns,AverErrFro(:,4),':*',Ns,AverErrFro(:,5),':+','linewidth',2);
legend('L1-ALM','L1-ADM','Lq-ADM (q=0.5)','Hard-ADM','SCAD-ADM','Location','Best');grid;xlim([Ns(1) Ns(end)]);
ylabel('Averaged relative error (Frobenius norm)'); xlabel('Number of samples (N)'); 

figure(4);subplot(1,2,2);
plot(Ns,AverErrSpe(:,1),'-',Ns,AverErrSpe(:,2),'--+',Ns,AverErrSpe(:,3),'-.',Ns,AverErrSpe(:,4),':*',Ns,AverErrSpe(:,5),':+','linewidth',2);
% legend('L1-ALM','L1-ADM','Lq-ADM (q=0.5)','Hard-ADM','SCAD-ADM','Location','Best');grid;xlim([Ns(1) Ns(end)]);
ylabel('Averaged relative error (Spectral norm)'); xlabel('Number of samples (N)'); grid;xlim([Ns(1) Ns(end)]);

