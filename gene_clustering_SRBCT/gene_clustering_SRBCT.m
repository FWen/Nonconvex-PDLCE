clear all; %close all;clc;

dataT = readtable('supplemental_data.txt','Delimiter','\t','ReadVariableNames',true);
data = cell2mat( cellfun(@str2double,dataT{2:end,3:65},'un',0) );

%EWS: 1:23,  23
%BL:  24:31, 8
%NB:  32:43, 12
%RMS: 44:63, 20

ns = [23, 8, 12, 20];
M = 4;%class number
N = size(data,2);

for k=1:size(data,1)
    ma = mean(data(k,:));     % overall mean
    mc = [mean(data(k,1:23)),mean(data(k,24:31)),mean(data(k,32:43)),mean(data(k,44:63))];%sample mean of each class
    vc = [var(data(k,1:23)),var(data(k,24:31)),var(data(k,32:43)),var(data(k,44:63))];%sample variance of each class
    
    F(k) = 1/3 * sum( ns.*(mc-ma).^2 )  / ( 1/(N-M)*sum((ns-1).*vc) ); 
end


[Fs I]=sort(F,'descend');
% plot(Fs);

gdata1 = data(I(1:40),:);          % top 40 genes
gdata2 = data([I(end-159:end)],:); % bottom 160 genes
gdata = [gdata1;gdata2];           % selected 200 genes

S1 = cov(gdata1.');
STD_S1 = diag(S1).^(0.5);
C1  = diag(1./STD_S1)*S1*diag(1./STD_S1);

figure(1);imagesc(abs(hca_order(C1)));set(gca,'xtick',[]);box off;

% gdata = [gdata1;gdata2]; % selected and ordered genes

d = size(gdata,1);
n = size(gdata,2);
zthresh = 1e-5;

S2 = cov(gdata.');%%sample correlation
STD_S2 = diag(S2).^(0.5);
C2  = diag(1./STD_S2)*S2*diag(1./STD_S2);
zerorat0 = length(find(abs(C2)<zthresh))/d/d;
figure(2);subplot(3,3,1); imagesc(abs(hca_order(C2)));
title(['Sample correlation' ' (' num2str(zerorat0*100, '%10.2f') '%)']);set(gca,'xtick',[]);box off;


epsilon = 1e-3; %lower bound for the eigenvalue

%--Soft-thresholding---------------
Xi1 = SOT_cov_est(gdata.');
Xic1 = diag(1./diag(Xi1).^(0.5))*Xi1*diag(1./diag(Xi1).^(0.5));
zerorat1 = length(find(abs(Xic1)<zthresh))/d/d;
figure(2);subplot(3,3,2); imagesc(abs(hca_order(Xic1)));
title(['Soft thresholding' ' (' num2str(zerorat1*100, '%10.2f') '%)']);set(gca,'xtick',[]);box off;


%--Hard-thresholding---------------
Xi2 = HT_cov_est(gdata.');
Xic2 = diag(1./diag(Xi2).^(0.5))*Xi2*diag(1./diag(Xi2).^(0.5));
zerorat1 = length(find(abs(Xic2)<zthresh))/d/d;
figure(2);subplot(3,3,3); imagesc(abs(hca_order(Xic2)));
title(['Hard thresholding' ' (' num2str(zerorat1*100, '%10.2f') '%)']);set(gca,'xtick',[]);box off;


%--SCAD-thresholding---------------
Xi3 = SCADT_cov_est(gdata.');
Xic3 = diag(1./diag(Xi3).^(0.5))*Xi3*diag(1./diag(Xi3).^(0.5));
zerorat1 = length(find(abs(Xic3)<zthresh))/d/d;
figure(2);subplot(3,3,4); imagesc(abs(hca_order(Xic3)));
title(['SCAD thresholding' ' (' num2str(zerorat1*100, '%10.2f') '%)']);set(gca,'xtick',[]);box off;


%--ALM algorithm with L1-norm penalty---------------
[X1] = L1_sparse_cov_est(gdata.',epsilon);
Xc1 = diag(1./diag(X1).^(0.5))*X1*diag(1./diag(X1).^(0.5));
zerorat1 = length(find(abs(Xc1)<zthresh))/d/d;
figure(2);subplot(3,3,5); imagesc(abs(hca_order(Xc1)));
title(['L1-ALM' ' (' num2str(zerorat1*100, '%10.2f') '%)']);set(gca,'xtick',[]);box off;


%--Proposed L1-ADM----------------------
[X2] = Lq_sparse_cov_est(gdata.',1,epsilon);
Xc2 = diag(1./diag(X2).^(0.5))*X2*diag(1./diag(X2).^(0.5));
zerorat2 = length(find(abs(Xc2)<zthresh))/d/d;
figure(2);subplot(3,3,6); imagesc(abs(hca_order(Xc2)));
title(['L1-ADM' ' (' num2str(zerorat2*100, '%10.2f') '%)']);set(gca,'xtick',[]);box off;


%--Proposed Lq-ADM (q=0.5)---------------
[X3] = Lq_sparse_cov_est(gdata.',0.5,epsilon);
Xc3 = diag(1./diag(X3).^(0.5))*X3*diag(1./diag(X3).^(0.5));
zerorat3 = length(find(abs(Xc3)<zthresh))/d/d;
figure(2);subplot(3,3,7); imagesc(abs(hca_order(Xc3)));
title(['Lq-ADM (q=0.5)' ' (' num2str(zerorat3*100, '%10.2f') '%)']);set(gca,'xtick',[]);box off;


%--- Proposed Hard-ADM --------------
[X4] = Lq_sparse_cov_est(gdata.',0,epsilon);
Xc4 = diag(1./diag(X4).^(0.5))*X4*diag(1./diag(X4).^(0.5));
zerorat4 = length(find(abs(Xc4)<zthresh))/d/d;
figure(2);subplot(3,3,8); imagesc(abs(hca_order(Xc4)));
title(['Hard-ADM' ' (' num2str(zerorat4*100, '%10.2f') '%)']);set(gca,'xtick',[]);box off;


%--- Proposed SCAD-ADM --------------
[X5] = SCAD_sparse_cov_est(gdata.',epsilon);
Xc5 = diag(1./diag(X5).^(0.5))*X5*diag(1./diag(X5).^(0.5));
zerorat5 = length(find(abs(Xc5)<zthresh))/d/d;
figure(2);subplot(3,3,9); imagesc(abs(hca_order(Xc5)));
title(['SCAD-ADM' ' (' num2str(zerorat5*100, '%10.2f') '%)']);set(gca,'xtick',[]);box off;


%--- eigen-value plots --------------
eigV(1,:)  = sort((eig(Xi1)),'descend');
eigV(2,:)  = sort((eig(Xi2)),'descend');
eigV(3,:)  = sort((eig(Xi3)),'descend');
eigV(4,:)  = sort((eig(X1)),'descend');
eigV(5,:)  = sort((eig(X2)),'descend');
eigV(6,:)  = sort((eig(X3)),'descend');
eigV(7,:)  = sort((eig(X4)),'descend');
eigV(8,:)  = sort((eig(X5)),'descend');
xlab = 120:200;
figure(5);
plot(xlab,eigV(1,xlab),'g-',xlab,eigV(2,xlab),'g--',xlab,eigV(3,xlab),'g:',...
    xlab,eigV(4,xlab),'r-',xlab,eigV(5,xlab),'r--',xlab,eigV(6,xlab),'r:',xlab,eigV(7,xlab),'b-',xlab,eigV(8,xlab),'b--','linewidth',1);
legend('Soft thresholding','Hard thresholding','SCAD thresholding','L1-ALM','L1-ADM','Lq-ADM (q=0.5)','Hard-ADM','SCAD-ADM',3);grid on;
ylabel('Eigenvalue');xlabel('Eigenvalue Index');ylim([-0.5 0.1]);
