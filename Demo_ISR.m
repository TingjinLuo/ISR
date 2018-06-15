% function Robust2Outlier_ISR()
% Written by Luo Tingjin
% Justify the capped l2-lp norm much more robust than l2-norm,l2-l1 norm
% 2015.04.30

% Loss + Regu
% Capped L2-L1 + L21
% L21L21
% LS+L21
% LS+L2

% clear all;
close all;
% Generate two moon
Num_PerC = 200;
Num_PC = 30;
Num_Outlier = 3;
[X,Y]=generate_two_moons(40,20, 10,Num_PerC,1,0);
LW = 2;
MS = 6;
% Disply the original data
% figure,plot(X(Y==1,1),X(Y==1,2),'rx','LineWidth',LW,'MarkerSize',MS); hold on;
%    plot(X(Y==2,1),X(Y==2,2),'b+','LineWidth',LW,'MarkerSize',MS); hold on;
%    axis tight;
%    axis equal;
% legend('Class One','Class Two')

% Randomly Select the labeled data points
Num_P = length(Y);
i_idx = find(Y == 1);
randid = randperm(length(i_idx));
id_n1 = i_idx(randid(1:Num_PC));
i_idx = find(Y == 2);
randid = randperm(length(i_idx));
id_n2 = i_idx(randid(1:Num_PC));

trainIdx = [id_n1;id_n2];
UnLabledIdx = setdiff(randperm(Num_P),trainIdx);
UnLabledIdx = UnLabledIdx';

Y_L = Y(trainIdx);
X_L = X(trainIdx,:);
Y_U = Y(UnLabledIdx);
X_U = X(UnLabledIdx,:);

% Disply the original labeled data
% figure,plot(X_L(Y_L==1,1),X(Y_L==1,2),'rx','LineWidth',LW,'MarkerSize',10); hold on;
% plot(X_L(Y_L==1,1),X_L(Y_L==1,2),'b+','LineWidth',LW,'MarkerSize',10); hold on;

figure,plot(X(id_n1,1),X(id_n1,2),'rx','LineWidth',LW,'MarkerSize',10); hold on;
plot(X(id_n2,1),X(id_n2,2),'b+','LineWidth',LW,'MarkerSize',10); hold on;

% Disply the original unlabeled data
plot(X(UnLabledIdx,1),X(UnLabledIdx,2),'k.','MarkerSize',20); hold on;
%%
% Add the outliers/noise into the origin data
Y_LNoise = Y_L;
i_idx = find(Y_LNoise == 1);
randid = randperm(length(i_idx));
id_n1 = i_idx(randid(1:Num_Outlier));
i_idx = find(Y_LNoise == 2);
randid = randperm(length(i_idx));
id_n2 = i_idx(randid(1:Num_Outlier));
Y_LNoise(id_n1) =2;
Y_LNoise(id_n2) =1;

plot(X_L(Y_LNoise==1,1),X_L(Y_LNoise==1,2),'ro','LineWidth',LW,'MarkerSize',MS); hold on;
plot(X_L(Y_LNoise==2,1),X_L(Y_LNoise==2,2),'bs','LineWidth',LW,'MarkerSize',MS); hold on;
axis tight;
axis equal;
legend('Data Labeled as Class One','Data Labeled as Class Two','Unlabeled Data Points','Outliers of Class One','Outliers of Class Two');

XT = [X_L;X_U];
% Label Propagation
T = TransLabelR(Y_LNoise); % labeled data points
T(end+1:end+length(UnLabledIdx),:)=0; % unlabeled data
[Num_P,nFea] = size(XT);
C_Intv = unique(Y_LNoise);
Num_C = length(C_Intv);
alpha = 0.35;
para.p = 1;
method =1;
gama = 1;
para.lambda =gama*sqrt(log(Num_C*nFea)/Num_P);
para.theta = 50*para.lambda*Num_C; k = 6;
[G_SRFS,b,obj,SemiFS_Cl2lp_idx,~,SRFS_Y,LP_FISR] = ISR(XT,T,k,para,method,alpha);
fig = figure;axes1 = axes('Parent',fig,'FontSize',14);
plot(XT(SRFS_Y==1,1),XT(SRFS_Y==1,2),'ro','LineWidth',LW,'MarkerSize',10); hold on;
plot(XT(SRFS_Y==2,1),XT(SRFS_Y==2,2),'bs','LineWidth',LW,'MarkerSize',10); hold on;
axis tight;
axis equal;
legend('Data Predicted as Class One','Data Predicted as Class Two');

sum(abs([Y_L;Y_U]-SRFS_Y))

