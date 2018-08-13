function [ Q, i ] = tensorPowerMethodAcc( U1, V1, U0, V0, spa, R, maxIter, tol )

[M, N] = size(spa{1});

curL = 0;
if(size(V1, 1) == N)
    curL = 1;
end

if(size(U1, 1) == M)
    curL = 2;
end

if(size(R, 2) == 0)
    i = 0;
    Q = zeros(size(U1, 1), 0);
    return;
end

switch(curL)
    case 1   
        % mode 1 unfold size [MxL N]
        Y = Afunc1acc_t(R, U1, V1, U0, V0, spa);
        [Q, ~] = qr(Y, 0);
        for i = 1:maxIter
            Y = Atfunc1acc_t(Q, U1, V1, U0, V0, spa);
            Y = Afunc1acc_t (Y, U1, V1, U0, V0, spa);
            
            [Qi, ~] = qr(Y, 0);
            
            delta = norm(Q(:,1) - Qi(:,1), 2); 
            Q = Qi;
            
            if(delta < tol)
                break;
            end
        end
    case 2
        % mode 2 unfold size [M NxL]
        Y = Afunc2acc_t(R, U1, V1, U0, V0, spa);
        [Q, ~] = qr(Y, 0);
        for i = 1:maxIter
            Y = Atfunc2acc_t(Q, U1, V1, U0, V0, spa);
            Y =  Afunc2acc_t(Y, U1, V1, U0, V0, spa);
            
            [Qi, ~] = qr(Y, 0);
            
            delta = norm(Q(:,1) - Qi(:,1), 2); 
            Q = Qi;
            
            if(delta < tol)
                break;
            end
        end
    otherwise
        disp('wrong mode seleceted \n');
end

end
