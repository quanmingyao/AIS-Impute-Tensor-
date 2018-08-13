function [U, S, V, pwIter, nnZ] = ApproxSVTacc_t(U1, V1, U0, V0, gradM, mode,...
    R, lambda, pwTol, pwIter)

[ Q, pwIter ] = tensorPowerMethodAcc( U1, V1, U0, V0, gradM, R, pwIter, pwTol);

switch(mode)
    case 1
        Z = Atfunc1acc_t(Q, U1, V1, U0, V0, gradM);
    case 2
        Z = Atfunc2acc_t(Q, U1, V1, U0, V0, gradM);
    otherwise
        disp('wrong mode seleceted \n');
end

[U, S, V] = mySVD(Z');
[U, S, V, nnZ] = filterSVT(U, S, V, lambda);
U = Q*(U*S);
end
