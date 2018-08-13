function [U, S, V, out] = PostProcess_t( D, U, V, S )

if(exist('S','var'))
    theta = cat(1, diag(S{1}), diag(S{2}));
else
    theta = randn(size(U{1}, 2) + size(U{2}, 2), 1);
end

L = length(D);
M = size(U{2},1);
N = size(V{1},1);

row = cell(L, 1);
col = cell(L, 1);
val = cell(L, 1);
for l = 1:L
    [row{l}, col{l}, val{l}] = find(D{l});
end

lb = -1e+9*ones(size(theta));
ub = +1e+9*ones(size(theta));

% max number of iterations
param.maxIter = 30;    
% max number of calling the function
param.maxFnCall = 1000;  
% tolerance of constraint satisfaction
param.relCha = 1e+5;      
% final objective function accuracy parameter
param.tolPG = 1e-2;   
% stored gradients
param.m = 2;

grad = cell(L, 1);
for l = 1:L
    grad{l} = sparse(row{l}, col{l}, val{l}, M, N);
end

callfunc = @(theta) bfgsCallBack( theta, row, col, val, U, V, grad );

[theta, obj, iter, numCall, flag] = lbfgsb(theta,lb,ub,callfunc, [], [], param);

S{1} = diag(theta(1:size(U{1},2)));
S{2} = diag(theta(length(theta) - size(U{2},2) + 1:end));

out.obj = obj;
out.iter = iter;
out.numCall = numCall;

fprintf('bfgs iter:%d , obj:%.4d \n', iter, obj);

end

%% ---------------------------------------------------------------
function [f, g] = bfgsCallBack(s, row, col, val, U, V, grad)

M = size(U{2}, 1);
N = size(V{1}, 1);
L = length(val);
Modes = 2;

% make up sparse part
for mode = 1:Modes
    switch(mode)
    case 1
        for l = 1:L
            Um = U{mode}(M*(l-1) + 1: M*l,:);
            
            Km = size(U{mode}, 2);
            Sm = s(1:Km);
            
            Um = Um*diag(Sm);
            partl = partXY_blas(Um', V{mode}', row{l}, col{l}, length(row{l}));
            
            val{l} = val{l} - partl';
        end
    case 2
        for l = 1:L
            Vm = V{mode}(N*(l-1) + 1: N*l,:);
            
            Km = size(V{mode}, 2);
            Sm = s(length(s) - Km + 1:end);
            
            Vm = Vm*diag(Sm);
            partl = partXY_blas(U{mode}', Vm', row{l}, col{l}, length(row{l}));
            
            val{l} = val{l} - partl';
        end
    otherwise
        disp('wrong mode seleceted \n');
    end
end

% function value
f = 0;
for l = 1:L
    f = f + (1/2)*sum(val{l}.^2);
end

% gradient
if nargout > 1
    for l = 1:L
        setSval(grad{l}, - val{l}, length(val{l}));
    end
    
    g  = zeros(size(s));
    for mode = 1:Modes
        switch(mode)
        case 1
            Km = size(U{1}, 2);
            
            for k = 1:Km
                gk = 0;
                for l = 1:L
                    Um = U{mode}(M*(l-1) + 1: M*l,k);
                    Vm = V{mode}(:,k);

                    gk = gk + Um'*grad{l}*Vm;
                end
                g(k) = gk;
            end
        case 2
            Km = size(V{2}, 2);
            
            for k = 1:Km
                gk = 0;
                
                for l = 1:L
                    Um = U{mode}(:,k);
                    Vm = V{mode}(N*(l-1) + 1: N*l,k);
                    
                    gk = gk + Um'*grad{l}*Vm;
                end
                g(length(g) - Km + k) = gk;
            end
        otherwise
            disp('wrong mode seleceted \n');
        end
    end
end

end