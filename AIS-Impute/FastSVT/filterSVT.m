function [U, S, V, nnS] = filterSVT(U, S, V, lambda)

S = diag(S);
S = S - lambda;
nnS = sum(S > 1e-6);
S = S(1:nnS);

U = U(:, 1:nnS);
V = V(:, 1:nnS);
S = diag(S);

end
