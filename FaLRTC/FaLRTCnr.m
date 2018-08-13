function [Y, out] =  FaLRTCnr(M, alpha, para, tstData)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Fast Low Rank Tensor Completion no relaxation version (FaLRTCnr)
% Time: 03/11/2012
% Reference: "Tensor Completion for Estimating Missing Values 
% in Visual Data", PAMI, 2012.
% Converge rate = 1/k^2
% min_{X} : \sum_i \alpha_i\|X_{i(i)}\|_* 
% s.t.         : X_\Omega = M_\Omega
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

factor = 2;
L = 1;
C = 0.5;

Omega = (M ~= 0);

% initialization
X = M;

Y = X;
Z = X;
B = 0;

N = ndims(M);
dim = size(M);
Gx = zeros(dim);

tmp = zeros(1, N);
for i = 1:N
    tmp(i) = max(SingularValue(Unfold(X, dim, i))) * alpha(i) * 0.4;
end

S = cell(N, 1);
obj = zeros(maxIter, 1);
Time = zeros(maxIter, 1);
RMSE = zeros(maxIter, 1);
Lmax = 10*sum(maxIter^factor ./ tmp);

for k = 1:maxIter
    timeFlag = tic;
    
    % update mu
    mu = tmp / k^factor;
    
    a2m = alpha.^2 ./ mu;
    ma = mu ./ alpha;
    
    Ylast = Y;
    % test L
    while (true)
        b = (1+sqrt(1+4*L*B)) / (2*L);
        X = b/(B+b) * Z + B/(B+b) * Ylast;
        % compute f'(x) namely "Gx" and f(x) namely "fx"
        Gx = Gx * 0;
        fx = 0;
        for i = 1 : N
            [temp, sigma2] = Truncate(Unfold(X, dim, i), ma(i));
            temp = Fold(temp, dim, i);
            Gx = Gx + a2m(i) * temp;
            fx = fx + a2m(i)*(sum(sigma2) - sum(max(sqrt(sigma2)-ma(i), 0).^2));
        end
        Gx(Omega) = 0;

        % compute f(Ytest) namely fy
        Y = X - Gx / L;
        fy = 0;
        for i = 1 : N
            [sigma] = SingularValue(Unfold(Y, dim, i));
            S{i} = sigma;
            fy = fy + a2m(i)*(sum(sigma.^2) - sum(max(sigma-ma(i), 0).^2));
        end
        % test if L(fx-fy) > |Gx|^2
        if (fx - fy)*L < sum(Gx(:).^2)
            if L < Lmax
                L = L/C;
            else
                L = Lmax;
                break;
            end
        else
             break;
        end
    end
    
    obj(k) = fy;
    if(k <= 1)
        delta = inf;
    else
        delta = abs(obj(k - 1) - obj(k))/obj(k);
    end
    
    % update Z, Y, and B
    Z = Z - b*Gx;
    B = B+b;
    
    if(exist('tstData', 'var'))
        RMSEk = Y - tstData;
        RMSEk = RMSEk.* (tstData ~= 0);
        RMSEk = sum(RMSEk(:).^2);
        RMSEk = sqrt(RMSEk/nnz(tstData));
        
        RMSE(k) = RMSEk;
    end
    
    if(k <= 1)
        Time(1) = toc(timeFlag);
    else
        Time(k) = Time(k - 1) + toc(timeFlag);
    end
    fprintf('FaLRTCnr iter:%d, obj:%.3d(%.2d)\n', k, obj(k), delta);
    if delta < tol
        break;
    end
end

out.S = S;
out.obj = obj(1:k);
out.Time = Time(1:k);
out.RMSE = RMSE(1:k);

end

