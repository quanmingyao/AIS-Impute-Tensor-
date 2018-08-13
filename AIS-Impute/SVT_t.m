function [U, S, V] = SVT_t(Z, lambda, temp)

[U, S, V] = svd(Z, 'econ');

S = diag(S);
S = max(S - lambda, 0);
S = S(1:nnz(S));

U = U(:,1:length(S));
V = V(:,1:length(S));
S = diag(S);

end

