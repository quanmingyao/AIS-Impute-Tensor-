function [U, S, V, out ] = SoftImpute_t( D, lambda, para )
% D: cell

if(isfield(para, 'maxIter'))
    maxIter = para.maxIter;
else
    maxIter = 5000;
end

if(isfield(para, 'tol'))
    tol = para.tol;
else
    tol = 1e-3;
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
clear D;

% initialize X = U S V = 0. 
curR = ones(Modes, 1);
[U, V] = initialX(M, N, L, curR);

gradM = cell(L, 1);
gradV = cell(L, 1);
for l = 1:L
    gradM{l} = sparse(row{l}, col{l}, val{l}, M, N);
    gradV{l} = val{l};
end

St = cell(Modes, 1);
for m = 1:Modes
    St{m} = 0;
end
obj = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);

for t = 1:maxIter
    ti = tic;
    
    % get objective
    objt = 0;
    for l = 1:L
        objt = objt + (1/2)*sum(gradV{l}.^2);
    end
    for m = 1:Modes
        objt = objt + lambda(m)*sum(St{m}(:));
    end
    obj(t) = objt;
    
    % convergence check
    if(t <= 1)
        delta = inf;
    else
        delta = (obj(t - 1) - obj(t))/obj(t);
    end
    
    fprintf('iter:%d, obj:%.3d(%.2d), rnk:(%d,%d) \n', ...
        t, objt, delta, nnz(St{1}), nnz(St{2}));
    
    % pick up one mode
    mode = mod(t - 1, Modes) + 1;
    
    % proximal step
    [Ut, St{mode}, Vt] = tensorSVD( U{mode}, V{mode}, gradM, curR(mode) );
    [Ut, St{mode}, Vt, nnZ] = filterSVT(Ut, St{mode}, Vt, lambda(mode));
    Ut = Ut*St{mode};
    
    if(curR(mode) <= nnZ)
        curR(mode) = curR(mode) + 1;
    else
        curR(mode) = nnZ + 1;
    end
    curR(mode) = min(curR(mode), maxR);
    
    % update gradient (remove old add new)
    switch(mode)
    case 1
        for l = 1:L
            Um = U{mode}(M*(l-1) + 1: M*l,:);
            part = partXY_blas(Um', V{mode}', row{l}, col{l}, length(row{l}));
            gradV{l} = gradV{l} + part';
            
            Um = Ut(M*(l-1) + 1: M*l,:);
            part = partXY_blas(Um', Vt', row{l}, col{l}, length(row{l}));
            gradV{l} = gradV{l} - part';
            
            setSval(gradM{l}, gradV{l}, length(gradV{l}));
        end
    case 2
        for l = 1:L
            Vm = V{mode}(N*(l-1) + 1: N*l,:);
            part = partXY_blas(U{mode}', Vm', row{l}, col{l}, length(row{l}));
            gradV{l} = gradV{l} + part';
            
            Vm = Vt(N*(l-1) + 1: N*l,:);
            part = partXY_blas(Ut', Vm', row{l}, col{l}, length(row{l}));
            gradV{l} = gradV{l} - part';
            
            setSval(gradM{l}, gradV{l}, length(gradM{l}));
        end
    otherwise
        disp('wrong mode seleceted \n');
    end
    
    U{mode} = Ut;
    V{mode} = Vt;
    
    if(t == 1)
        Time(t) = toc(ti);
    else
        Time(t) = Time(t - 1) + toc(ti);
    end
    
    if(isfield(para, 'test'))        
        RMSE(t) = AISPredictSparse( U, V, para.test );
        fprintf('testing RMSE:%.2d \n', RMSE(t));
    end
    
    if(abs(delta) < tol)
        break;
    end
end

% make orthorgnal output
[U, S, V] = MakeOrthOut(U, V);

out.S = S;
out.obj = obj(1:t);
out.Time = Time(1:t);
out.RMSE = RMSE(1:t);

end

%% --------------------------------------------------------------
function [U, V] = initialX(M, N, L, rnk)

U{1} = zeros(M*L, rnk(1));
V{1} = zeros( N , rnk(1));

U{2} = zeros( M , rnk(2));
V{2} = zeros(N*L, rnk(2));

end

