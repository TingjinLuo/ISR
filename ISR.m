function [W,b,obj,feature_idx,Error_W,Y,F] = ISR(X_tr,T,k,para,method,alpha)
% function [W,b,feature_idx] = ISR(X_tr,T,k,para,method,alpha)
% Semi supervied feature selection via capped L2L1 regularization
% Input:
% X_tr: Rows of vectors of data points. Each row is x_i
% T: n*(c+1) class indicator matrix. Tij=1 if xi is labeled as j, Tij=0 otherwise
% k: the number of KNNs of Laplacian graph
% Para:
%       lambda: control regular term parameter
%       p: l2-lp norm,0<p<=1
%       theta: threshold parameter
% alpha: parameter associated with labeled data points
% method: 1:l2-lp,2:capped l2-lp
% Output:
% W: the learned d*c projection matrix
% b: the learned c*1 bias vector
% F: the n*(c+1) virtural label matrix
% proF: predicted n*c label matrix via W and b
% Y: predicted label vector via W and b
% obj: objective function value
% feature_idx: feature index by importance of each feature

% Objective function: (1) min trace(F'LF)+trace[(F-Y)'U(F-Y)]
% (2) min_W L(W) + \sum_i r_i(W)
% ================================ loss function ==========================
% L(W) = 1/2 \sum_i \|X_i*w_i - y_i\|^2
% ================================ regularizer ============================
% Capped L1 trace regularizer (CapL1 trace)
% r_i(W) = lambda* sum_j[min(||x^j||_p,theta)], (theta > 0, lambda >= 0)


% W = ISR(X_tr, T, k,alpha,lambda,p, method);
% Ref:
% 2014.05.25--05.26
%  Written by LUO Tingjin

p = para.p;
theta = para.theta;
lambda = para.lambda;

Num_Iter =60; % Number of method's main loop

if nargin < 5
    method =1;
end;
if nargin >5
    flag =0;
else
    flag =1;
end
% (1) Label Propagation via Enhanced Random Walks
% Compute the Laplacian matrix
[nSmp,nFea] = size(X_tr);
options = [];
options.t = 1;
options.NeighborMode = 'KNN';
options.k = k;
options.WeightMode = 'HeatKernel';
A= constructW_DC(X_tr,options);
D = full(sum(A,2));
D = sparse(1:nSmp,1:nSmp,D,nSmp,nSmp);
L = D-A;
Num_C = size(T,2);
% Compute Compute soft label matrix F or Fc
MaxC = 10000;
switch flag
    case 1
        % via min trace(F'LF)+trace[(F-Y)'U(F-Y)]
        U = zeros(nSmp,1)+MaxC;
        UnlabeledIdx = find(sum(T,2)==0);
        U(UnlabeledIdx) = eps;
        U = sparse(1:nSmp,1:nSmp,U,nSmp,nSmp);
        F = (L+U)\U*T;
    case 0
        % Compute soft label matrix F or Fc via Enhanced Random Walks
        D = sum(A,2);
        invD  = spdiags(1./D,0,nSmp,nSmp);
        P = invD*A;
        labeled_idx = find(sum(T,2) == 1);
        t = ones(nSmp,1);
        t(labeled_idx) = 0;
        T = [T, t];
        
        %  G = (I-Ia*P)^-1*Ib
        I_a = alpha*ones(nSmp,1);
        I_a(labeled_idx) = 0;
        I_a = spdiags(I_a,0,nSmp,nSmp);
        I_b = speye(nSmp) - I_a;
        H = speye(nSmp) - I_a * P;
        
        %F: n*(c+1) soft label matrix, F = GY
        F = H\(I_b*T);
end
Fc = F(:,1:Num_C); %F(:,1:end-1)

% Regularized  Regression

switch method
    case 1
        % l2-lp norm regulariser
        % Initialize D = I and lambda;
        D = []; D = eye(nFea);
        % lambda = ones(nFea,1)*lambda;
        for i=1:Num_Iter
            sv = sum(Fc,2);
            S = sparse(diag(sv));
            s = S*sparse(ones(nSmp,1));
            
            Ts = X_tr'*S*X_tr;  aa = X_tr'*s; Ts1 = (1/sum(sv)*aa)*aa'; St = Ts-Ts1;
            B = (X_tr'- 1/sum(sv)*aa*ones(1,nSmp))*Fc;
            
            gammapara = lambda;
            W = (St+gammapara*D)\B;
            
            pro = X_tr*W;
            b = 1/sum(sv)*ones(1,nSmp)*(Fc - S*pro);
            b = b';
            
            % update D
            % ${d_k} = \frac{p}{2}{({\left\| {{w^k}} \right\|^2} + \varepsilon )^{\frac{{p - 2}}{2}}}$
            di = 0.5*power(sum(W.*W,2)+eps,0.5*p-1);
            D = diag(di);
            obj(i) = trace((W'*X_tr'+b*ones(1,nSmp))*S*(X_tr*W+ones(nSmp,1)*b'))-2*trace(Fc*(W'*X_tr'+b*ones(1,nSmp)))+sum(sv)+gammapara*trace(W'*D*W);
            if i==1
                Error_W(i)= sum(sqrt(sum(W.*W,2)));
                tempW = W;
            else
                Error_W(i) = sum(abs(sqrt(sum(W.*W,2))-sqrt(sum(tempW.*tempW,2))));
                tempW = W;
            end
        end
    case 2
        % capped l2-lp norm regulariser
        % Initialize D = I and lambda;
        D = []; D = eye(nFea);
        lambda = ones(nFea,1)*lambda;
        for i=1:Num_Iter
            sv = sum(Fc,2);
            S = sparse(diag(sv));
            s = S*sparse(ones(nSmp,1));
            
            Cs = eye(nSmp)-1/sum(sv)*s*sparse(ones(1,nSmp));
            Ls = Cs*S;
            
            % Ts = X_tr'*S*X_tr;  aa = X_tr'*s; Ts1 = (1/sum(sv)*aa)*aa'; St = Ts-Ts1;
            % B = (X_tr'- 1/sum(sv)*aa*ones(1,nSmp))*Fc;
            
            gammapara = diag(lambda);
            % W = (St+gammapara*D)\B;
            W = (X_tr'*Ls*X_tr+gammapara*D)\X_tr'*Cs*Fc;
            
            pro = X_tr*W;
            b = 1/sum(sv)*ones(1,nSmp)*(Fc - S*pro);
            b = b';
            
            % update D
            % ${d_k} = \frac{p}{2}{({\left\| {{w^k}} \right\|^2} + \varepsilon )^{\frac{{p - 2}}{2}}}$
            di = 0.5*power(sum(W.*W,2)+eps,0.5*p-1);
            Ind_T = find(sum(W.*W,2)+eps > theta);
            di(Ind_T) = 0;
            lambda(Ind_T) =0;
            D = diag(di);
            
            obj(i) = trace((W'*X_tr'+b*ones(1,nSmp))*S*(X_tr*W+ones(nSmp,1)*b'))-2*trace(Fc*(W'*X_tr'+b*ones(1,nSmp)))+sum(sv)+trace(W'*gammapara*D*W);
            if i==1
                Error_W(i)= sum(sqrt(sum(W.*W,2)));
                tempW = W;
            else
                Error_W(i) = sum(abs(sqrt(sum(W.*W,2))-sqrt(sum(tempW.*tempW,2))));
                tempW = W;
            end
        end
end

score = sum(W.*W,2);
[~, feature_idx] = sort(score,'descend');

proF = X_tr*W + ones(nSmp,1)*b';
[dumb, Y] = max(proF,[],2);
