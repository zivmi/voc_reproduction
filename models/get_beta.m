function B = get_beta(Y,X,lambda_list)

% Y       Tx1
% X       TxP
% lambda  Lx1
% B       PxL

% % Uncomment to check code
% T       = 100;
% P       = 1000;
% X       = randn(T,P);
% Y       = randn(T,1);
% lambda  = 1;

% Ytrn',Ztrn',lamlist

% for test
% Y = Ytrn';
% X = Ztrn';
% lambda_list = lamlist;

if sum(isnan(X(:)))+sum(isnan(Y))>0
    error('missing data')
end

L_       = length(lambda_list);
T_ = size(X,1);
P_  = size(X,2);

if P_ > T_
    a_matrix = X*X'/T_; % T_ \times T_
else
    a_matrix = X'*X/T_; % P_ \times P_
end

[U_a,D_a,V_a] = svd(a_matrix);

scale_eigval = diag((D_a*T_).^(-1/2));

W = X'*U_a* diag(scale_eigval);

a_matrix_eigval = diag(D_a);

signal_times_return = X'*Y / T_;  % This is (SR): M \times 1
signal_times_return_times_v = W' * signal_times_return;  % this is V' * (SR) # T \times 1

B       = nan(P_,L_);
for l=1:L_
    B(:,l)      = W * diag(1./(a_matrix_eigval + lambda_list(l))) * signal_times_return_times_v; 
end

end

