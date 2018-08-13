function [U, S, V] = mySVD(Z)

[m, n] = size(Z);

if(m >= n)
    [U, S, V] = svd(Z, 'econ');
else
    [U, S, V] = svd(Z', 'econ');
    temp = U;
    U = V;
    V = temp;
end

end