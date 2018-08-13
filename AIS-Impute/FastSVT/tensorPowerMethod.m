function [ Q, i ] = tensorPowerMethod( U, V, spa, R, maxIter, tol )

[M, N] = size(spa{1});

curL = 0;
if(size(V, 1) == N)
    curL = 1;
end

if(size(U, 1) == M)
    curL = 2;
end

switch(curL)
    case 1   
        % mode 1 unfold size [MxL N]
        Y = Afunc1(R, U, V, spa);
        [Q, ~] = qr(Y, 0);
        for i = 1:maxIter
            Y = Atfunc1(Q, U, V, spa);
            Y =  Afunc1(Y, U, V, spa);
            
            [Qi, ~] = qr(Y, 0);
            
            delta = norm(Q(:,1) - Qi(:,1), 2); 
            Q = Qi;
            
            if(delta < tol)
                break;
            end
        end
    case 2
        % mode 2 unfold size [M NxL]
        Y = Afunc2(R, U, V, spa);
        [Q, ~] = qr(Y, 0);
        for i = 1:maxIter
            Y = Atfunc2(Q, U, V, spa);
            Y =  Afunc2(Y, U, V, spa);
            
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
