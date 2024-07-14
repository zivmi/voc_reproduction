function B = ridgesvd(Y,X,lambda)

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

% Y = Ytrn';
% X = Ztrn';
% lambda = lamlist/trnwin;

% Ytrn',Ztrn',lamlist
if sum(isnan(X(:)))+sum(isnan(Y))>0
    error('missing data')
end

L       = length(lambda);
% [U,S,V] = svd(A) performs a singular value decomposition of matrix A, such that A = U*S*V'.
[U,D,V] = svd(X);
D       = diag(D);
[T,P]   = size(X);

if T>=P
    compl   = zeros(P,T-P);
else
    compl   = zeros(P-T,T);
end
B       = nan(P,L);

for l=1:L
    if T>=P
        B(:,l)      = V*[diag(D./(D.^2+lambda(l))),compl]*U'*Y;
    else
        B(:,l)      = V*[diag(D./(D.^2+lambda(l)));compl]*U'*Y;
    end   
    
end

% % Uncomment to check code
% scatter45line(B(:,1),(X'*X+lambda(1)*eye(P))\X'*Y)

