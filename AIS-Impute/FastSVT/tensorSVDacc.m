function [ U, S, V ] = tensorSVDacc( U1, V1, U0, V0, spa, rnk)

[M, N] = size(spa{1});
L = length(spa);

curL = 0;
if(size(V1, 1) == N)
    curL = 1;
end

if(size(U1, 1) == M)
    curL = 2;
end

switch(curL)
    case 1   
        % mode 1 unfold size [MxL N]
        Afunc  = @(x) Afunc1acc_t (x, U1, V1, U0, V0, spa);
        Atfunc = @(x) Atfunc1acc_t(x, U1, V1, U0, V0, spa);

        [U, S, V] = lansvd(Afunc, Atfunc, M*L, N, rnk, 'L');
    case 2
        % mode 2 unfold size [M NxL]
        Afunc  = @(x) Afunc2acc_t (x, U1, V1, U0, V0, spa);
        Atfunc = @(x) Atfunc2acc_t(x, U1, V1, U0, V0, spa);

        [U, S, V] = lansvd(Afunc, Atfunc, M, N*L, rnk, 'L');
    otherwise
        disp('wrong mode seleceted \n');
end

end
