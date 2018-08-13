function [U, S, V] = tensorSVD( U, V, spa, rnk )

[M, N] = size(spa{1});
L = length(spa);

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
        Afunc = @(x) Afunc1_t(x, U, V, spa);
        Atfunc = @(x) Atfunc1_t(x, U, V, spa);

        [U, S, V] = lansvd(Afunc, Atfunc, M*L, N, rnk, 'L');
    case 2
        % mode 2 unfold size [M NxL]
        Afunc = @(x) Afunc2_t(x, U, V, spa);
        Atfunc = @(x) Atfunc2_t(x, U, V, spa);

        [U, S, V] = lansvd(Afunc, Atfunc, M, N*L, rnk, 'L');
    otherwise
        disp('wrong mode seleceted \n');
end

end