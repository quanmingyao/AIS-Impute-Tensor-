function [ Y, out ] = APG_t3( D, lambda, para )

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

Omega = (D ~= 0);
sz = size(D);
Modes = length(sz);

X0 = cell(Modes, 1);
X1 = cell(Modes, 1);
Y  = cell(Modes, 1);
for m = 1:Modes
    X0{m} = zeros(sz);
    X1{m} = zeros(sz);
     Y{m} = zeros(sz);
end

S = cell(Modes, 1);
curR = ones(Modes, 1);

a0 = 1;
a1 = 1;
obj = zeros(maxIter, 1);
Time = zeros(maxIter, 1);

for i = 1:maxIter 
    ti = tic;
    
    theta = (a0 - 1)/a1;
    % theta = 0;
    
    % proximal step
    for m = 1:Modes
        Y{m} = X1{m} + theta*(X1{m} - X0{m});
    end
    clear m;
    
    grad = -D;
    for m = 1:Modes
        grad = grad + (Y{m}.*Omega);
    end
    clear m;

    stepsize = 2;

    for mode = 1:Modes
        Zm = Y{mode} - grad/stepsize;
        Zm = Unfold(Zm, sz, mode);
        [U{mode}, S{mode}, V{mode}] = SVT_t(Zm, lambda(mode)/stepsize, curR(mode));
        Zm = U{mode}*S{mode}*V{mode}';

        X0{mode} = X1{mode};
        X1{mode} = Fold(Zm, sz, mode);
        S{mode} = diag(S{mode});

        % update rank
        nnZ = nnz(S{mode});
        if(curR(mode) <= nnZ)
            curR(mode) = curR(mode) + 1;
        else
            curR(mode) = nnZ + 1;
        end
        curR(mode) = min(curR(mode), maxR);
    end

    % object value
    grad = -D;
    for m = 1:Modes
        grad = grad + (X1{m}.*Omega);
    end
    clear m;
    obji = (1/2)*sum(grad(:).^2);
    for m = 1:Modes
        obji = obji + lambda(m)*sum(S{m});
    end
    clear m;
    obj(i) = obji;    

    if(i <= 1)
        delta = inf;
    else
        delta = (obj(i - 1) - obj(i))/obj(i); 
    end
    
    if(delta < 0)
        a0 = 1;
        a1 = 1;
        fprintf('restart \n');
    else
        at = (1 + sqrt(1 + 4*a1^2))/2;
        a0 = a1;
        a1 = at;
    end
        
    if(i == 1)
        Time(i) = toc(ti);
    else
        Time(i) = Time(i - 1) + toc(ti);
    end
    
    fprintf('iter:%d, obj:%.2d(%.1d), mode:%d, rank:(%d,%d,%d), lambda:%.2d\n',...
        i, obji, delta, mode, [nnz(S{1}),nnz(S{2}),nnz(S{3})], lambda(mode));
    if(delta < tol && delta > 0)
        break;
    end
end

Y = zeros(size(D));
for m = 1:Modes
    Y = Y + X1{m};
end

out.S = S;
out.obj = obj(1:i);
out.Time = Time(1:i);
out.Y = Y;

end

