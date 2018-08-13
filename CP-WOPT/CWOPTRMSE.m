function [ RMSE ] = CWOPTRMSE( A, testData )

m = size(A{1}, 2);
m = m*ones(1, length(A));
C = zeros(m);
for i = 1:size(C,1)
    C(i,i,i) = 1;
end
C = tensor(C);
A = ttensor(C, A);

RMSE = sqrt(2*calcFunction( testData, A)/nnz(testData));

end

