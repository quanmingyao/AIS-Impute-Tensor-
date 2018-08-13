function [ U1, S, V1, out ] = AISImpute_t( D, lambda, para )
% D: cell
% mode 1 [ml x n]; mode 2 [m x nl]

if(isfield(para, 'maxIter'))
    maxIter = para.maxIter;
else
    maxIter = 5000;
end

if(~isfield(para, 'tol'))
    para.tol = 1e-3;
end

if(isfield(para, 'maxR'))
    maxR = para.maxR;
else
    maxR = 100;
end

% mode 3 is not low rank
[M, N] = size(D{1});
L = length(D);
Modes = 2;

row = cell(L, 1);
col = cell(L, 1);
val = cell(L, 1);
for l = 1:L
    [row{l}, col{l}, val{l}] = find(D{l});
end

% lambdaMax = getLambdaMax(D);

clear D;

% initialize X = U S V = 0. 
curR = ones(Modes, 1);

[U1, V1, U0, V0, St] = initialX(M, N, L, curR);
[part0, part1, gradM] = initialParts(row, col, val, M, N, L);

a0 = 1;
a1 = 1;
obj = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);

for t = 1:maxIter
    tt = tic;
    
    theta = (a0 - 1)/a1;
    
    % larger step size can be adopted at first few iterations
%     stepsize = 0.5 * t/(t + 1);
    stepsize = sqrt(2);
    
    % make up sparse part
    for l = 1:L
        partl = (1 + theta)*part1{1, l} - theta*part0{1, l};
        partl = (1 + theta)*part1{2, l} - theta*part0{2, l} + partl;
        partl = val{l} - partl;
        
        setSval(gradM{l}, partl/stepsize, length(partl));
    end
    
    % pick up one mode (cicle is best)
    for mode = 1:Modes
        % mode = mod(t - 1, Modes) + 1;
        lambdai = lambda(mode);

        % proximal step
        [ R, ~ ] = qr( [V1{mode}, V0{mode}], 0 );
        R = R(:, 1:min(maxR, size(R, 2)));
        if(para.exact == 1)
            [Ut, St{mode}, Vt] = tensorSVDacc( (1 + theta)*U1{mode}, V1{mode}, ...
                (-theta)*U0{mode}, V0{mode}, gradM, curR(mode) );
            [Ut, St{mode}, Vt] = filterSVT(Ut, St{mode}, Vt, lambdai/stepsize);
            Ut = Ut*St{mode};

            pwTol = 0; pwIter = inf;
        else
            pwTol = 0;

            [Ut, St{mode}, Vt, pwIter] = ApproxSVTacc_t((1 + theta)*U1{mode}, ...
                V1{mode}, (-theta)*U0{mode}, V0{mode}, gradM, mode, R, ...
                lambdai/stepsize, 0, 3);
        end
        U0{mode} = U1{mode};
        U1{mode} = Ut;
        V0{mode} = V1{mode};
        V1{mode} = Vt;

        curR(mode) = min(size(R, 2), maxR);
    end
    
    % make up and update sparse part
    objt = 0;
    for m = 1:Modes
        objt = objt + lambda(m)*sum(St{m}(:));
    end
    
    % update sparse part and get object value
    for l = 1:L
        Um = U1{1}(M*(l-1) + 1: M*l,:);
        partl = partXY_blas(Um', V1{1}', row{l}, col{l}, length(row{l}));

        part0{1,l} = part1{1,l};
        part1{1,l} = partl';

        Vm = V1{2}(N*(l-1) + 1: N*l,:);
        partl = partXY_blas(U1{2}', Vm', row{l}, col{l}, length(row{l}));

        part0{2,l} = part1{2,l};
        part1{2,l} = partl';

        partl = val{l} - part1{1,l} - part1{2,l};

        objt = objt + (1/2)*sum(partl.^2);
    end

    obj(t) = objt;
    if(t <= 1)
        delta = inf;
    else
        delta = (obj(t - 1) - obj(t))/(obj(t));
    end
    
    % adaptive restart
    if(delta < 0)
        a0 = 1;
        a1 = 1;
    else
        at = (1 + sqrt(1 + 4*a1^2))/2;
        a0 = a1;
        a1 = at;
    end
    
    if(t == 1)
        Time(t) = toc(tt);
    else
        Time(t) = Time(t - 1) + toc(tt);
    end
    
    fprintf('iter:%d, obj:%.2d(%d,%.1d), rnk:in(%d,%d) out(%d,%d), power:(%d,%.2d)\n', ...
        t, obj(t), mode, delta, curR(1), curR(2), ...
        nnz(St{1}), nnz(St{2}), pwIter, pwTol);
    
    if(isfield(para, 'test'))        
        RMSE(t) = AISPredictSparse( U1, V1, para.test );
        fprintf('testing RMSE:%.2d \n', RMSE(t));
    end
    
    if(delta > 0 && delta < para.tol)
        break;
    end
end

% make orthorgnal output
[U1, S, V1] = MakeOrthOut(U1, V1);

out.S = S;
out.obj = obj(1:t);
out.Time = Time(1:t);
out.RMSE = RMSE(1:t);

end

% %% --------------------------------------------------------------
% function [lambda] = getLambdaMax(data)
% 
% [~, N, L] = cat(2, size(data{1}), length(data));
% lambda = zeros(L, 1);
% 
% r = randn(N, 1);
% for l = 1:L
%     part = data{l};
%     
%     q = powerMethod( part, r, 5, 1e-6);
%     s = q'*part;
%     
%     lambda(l) = norm(s, 2);
% end
% 
% lambda = mean(lambda);
% 
% end

%% --------------------------------------------------------------
function [U1, V1, U0, V0, S] = initialX(M, N, L, rnk)

U0{1} = zeros(M*L, rnk(1));
V0{1} = randn( N , rnk(1));

U0{2} = zeros( M , rnk(2));
V0{2} = randn(N*L, rnk(2));

U1{1} = zeros(M*L, rnk(1));
V1{1} = randn( N , rnk(1));

U1{2} = zeros( M , rnk(2));
V1{2} = randn(N*L, rnk(2));

Modes = 2;
S = cell(Modes, 1);
for m = 1:Modes
    S{m} = 0;
end

end

%% --------------------------------------------------------------
function [part0, part1, gradM] = initialParts(row, col, val, M, N, L)

Modes = 2;

part0 = cell(Modes, L);
part1 = cell(Modes, L);
gradM = cell(L, 1);
for l = 1:L
    part0{1, l} = zeros(size(val{l}));
    part1{1, l} = part0{1, l};
    
    part0{2, l} = zeros(size(val{l}));
    part1{2, l} = part0{2, l};
    
    gradM{l} = sparse(row{l}, col{l}, val{l}, M, N);
end

end

