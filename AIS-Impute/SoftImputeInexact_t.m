function [ out ] = SoftImputeInexact_t( D, lambda, para )
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

% lambdaMax = getLambdaMax(D);

clear D;

% initialize X = U S V = 0. 
curR = ones(Modes, 1);
[U, V1] = initialX(M, N, L, curR);

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
for t = 1:maxIter
    lambdai = lambda;
    
    % get objective
    objt = 0;
    for l = 1:L
        objt = objt + (1/2)*sum(gradV{l}.^2);
    end
    for m = 1:Modes
        objt = objt + lambda*sum(St{m}(:));
    end
    obj(t) = objt;
    
    % pick up one mode
    mode = mod(t - 1, Modes) + 1;
    
    % proximal step
    R = [V1{mode}, V1{mode}];
    [R, ~] = qr(R, 0);
    R = R(:, 1:curR(mode));
    
    pwTol = max(1e-2*0.99^(t - 1), 1e-4);
    [ Q, pwIter ] = tensorPowerMethod( U{mode}, V1{mode}, gradM, R, 3, pwTol);
    
    switch(mode)
        case 1
            Z = Atfunc1(Q, U{mode}, V1{mode}, gradM);
        case 2
            Z = Atfunc2(Q, U{mode}, V1{mode}, gradM);
        otherwise
            disp('wrong mode seleceted \n');
    end
    
    [Ut, St{mode}, Vt] = mySVD(Z');
    Ut = Q*Ut;
    
    [Ut, St{mode}, Vt, nnZ] = filterSVT(Ut, St{mode}, Vt, lambdai);
    Ut = Ut*St{mode};
    
    if(curR(mode) <= nnZ)
        curR(mode) = curR(mode) + 1;
    else
        curR(mode) = nnZ + 1;
    end
    
    % convergence check
    if(t <= 1)
        delta = inf;
    else
        delta = obj(t - 1) - obj(t);
    end
    
    fprintf('iter:%d, obj:%.3d(%.2d), rnk:(%d,%d), power:(%d, %.2d) \n', ...
        t, objt, delta, curR(1), curR(2), pwIter, pwTol);
    if(abs(delta) < tol && abs(delta) > 1e-10)
        break;
    end
    
    % update gradient (remove old add new)
    switch(mode)
    case 1
        for l = 1:L
            Um = U{mode}(M*(l-1) + 1: M*l,:);
            part = partXY_blas(Um', V1{mode}', row{l}, col{l}, length(row{l}));
            gradV{l} = gradV{l} + part';
            
            Um = Ut(M*(l-1) + 1: M*l,:);
            part = partXY_blas(Um', Vt', row{l}, col{l}, length(row{l}));
            gradV{l} = gradV{l} - part';
            
            setSval(gradM{l}, gradV{l}, length(gradV{l}));
        end
    case 2
        for l = 1:L
            Vm = V1{mode}(N*(l-1) + 1: N*l,:);
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
    V0{mode} = V1{mode};
    V1{mode} = Vt;
end

out.obj = obj(1:t);

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
function [U, V] = initialX(M, N, L, rnk)

U{1} = zeros(M*L, rnk(1));
V{1} = randn( N , rnk(1));

U{2} = zeros( M , rnk(2));
V{2} = randn(N*L, rnk(2));

end


