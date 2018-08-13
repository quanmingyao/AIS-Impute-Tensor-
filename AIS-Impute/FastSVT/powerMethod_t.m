function [Q, maxIter] = powerMethod_t( A, R, maxIter, tol)
% basic version of power method

if(~exist('tol', 'var'))
    tol = 1e-5;
end

Y = A*R;
[Q, ~] = qr(Y, 0);
err = zeros(maxIter, 1);
for i = 1:maxIter
    Y = A*(A'*Q);
    [iQ, ~] = qr(Y, 0);
    
    err(i) = norm(iQ(:,1) - Q(:,1), 1);
    Q = iQ;
    
    if(err(i) < tol)
        break;
    end
end

maxIter = i;

end

