clear; clc;

maxNumCompThreads(1);

for repeat = 1:10

factor_dims = 125;
factor_dims = [factor_dims, factor_dims, 3];
core_dims   = [3, 3, 3];
    
% Problem setting
U = cell(1, length(core_dims));
for i = 1:length(core_dims)
    U{i} = randn(factor_dims(i), core_dims(i));
    % [U{i}, ~] = qr(U{i}, 0);
end
C = randn(core_dims);

% Generate low-rank tensor
gnd = ttensor(tensor(C), U);

gnd = double(gnd);
gnd = gnd - mean(gnd(:));

X = gnd + randn(size(gnd))*mean(abs(gnd(:)))*0.05;

ratio = 5*prod(core_dims)/sqrt(prod(factor_dims));
O = rand(size(X));
idx = find(O < ratio);

M = zeros(size(O));
M(idx) = 1;

clear O idx;

traX = cell(size(M, 3), 1);
for i = 1:size(M, 3)
    traX{i} = sparse(X(:,:,i).*M(:,:,i));
end

% AIS-Impute

para.tol = 1e-4;
para.maxIter = 5000;
para.exact = 0;
para.maxR = 6;

lambdaGrad = 10*(0.8).^(1:10);
for i = 1:length(lambdaGrad)
    lambda = [lambdaGrad(i), lambdaGrad(i)];
    
    [U, S, V, outGrid{i}] = AISImpute_t( traX, lambda, para );

    temp = AISPredictFull(U, S, V, length(traX));
    temp = (gnd - temp).*(1 - M);
    GridNMSE(1, i) = norm(temp(:), 2)/norm(gnd(:).*(1 - M(:)));
    
    [U, S, V] = PostProcess_t( traX, U, V, S);

    temp = AISPredictFull(U, S, V, length(traX));
    temp = (gnd - temp).*(1 - M);
    GridNMSE(2, i) = norm(temp(:), 2)/norm(gnd(:).*(1 - M(:)));
    
%     if(i > 1 && GridNMSE(2, i) > GridNMSE(2, i - 1))
%         break;
%     end
end

[~, lambda] = min(GridNMSE(2, 1:i));
lambda = [lambdaGrad(lambda), lambdaGrad(lambda)];

clear U S V temp lambdaGrad outBFGS GridNMSE i;

para.tol = 1e-6;
[~,~,~, minObj] = AISImpute_t( traX, lambda, para );

%% ------------------------------------------------------------------------
para.tol = 1e-5;

method = 1;

D = zeros(size(M));
for i = 1:length(traX)
    D(:,:,i) = full(traX{i});
end

[~, ~, ~, out{method}] = APG_t( D, lambda, para );

temp = (gnd - out{method}.Y).*(1 - M);
NMSE(1, method) = norm(temp(:), 2)/norm(gnd(:).*(1 - M(:)));

Time(1, method) = out{method}.Time(end);

clear D;

%% ------------------------------------------------------------------------
method = method + 1;
para.exact = 1;

[U, S, V, out{method}] = SoftImpute_t( traX, lambda, para );

temp = AISPredictFull(U, S, V, length(traX));
temp = (gnd - temp).*(1 - M);
NMSE(1, method) = norm(temp(:), 2)/norm(gnd(:).*(1 - M(:)));

Time(1, method) = out{method}.Time(end);

tt = tic;
[U, S, V] = PostProcess_t( traX, U, V, S);
Time(2, method) = toc(tt);

temp = AISPredictFull(U, S, V, length(traX));
temp = (gnd - temp).*(1 - M);
NMSE(2, method) = norm(temp(:), 2)/norm(gnd(:).*(1 - M(:)));

%% ------------------------------------------------------------------------
method = method + 1;
para.exact = 1;

[U, S, V, out{method}] = AISImpute_t( traX, lambda, para );

temp = AISPredictFull(U, S, V, length(traX));
temp = (gnd - temp).*(1 - M);
NMSE(1, method) = norm(temp(:), 2)/norm(gnd(:).*(1 - M(:)));

Time(1, method) = out{method}.Time(end);

tt = tic;
[U, S, V] = PostProcess_t( traX, U, V, S);
Time(2, method) = toc(tt);

temp = AISPredictFull(U, S, V, length(traX));
temp = (gnd - temp).*(1 - M);
NMSE(2, method) = norm(temp(:), 2)/norm(gnd(:).*(1 - M(:)));

%% ------------------------------------------------------------------------
method = method + 1;
para.exact = 0;

[U, S, V, out{method}] = AISImpute_t( traX, lambda, para );

temp = AISPredictFull(U, S, V, length(traX));
temp = (gnd - temp).*(1 - M);
NMSE(1, method) = norm(temp(:), 2)/norm(gnd(:).*(1 - M(:)));

Time(1, method) = out{method}.Time(end);

tt = tic;
[U, S, V] = PostProcess_t( traX, U, V, S);
Time(2, method) = toc(tt);

temp = AISPredictFull(U, S, V, length(traX));
temp = (gnd - temp).*(1 - M);
NMSE(2, method) = norm(temp(:), 2)/norm(gnd(:).*(1 - M(:)));

clear C gnd i lambda M outGrid para S traX tt U V X ans;

save(strcat('syn-', num2str(factor_dims(1)), '-', num2str(repeat), '.mat'));

end



