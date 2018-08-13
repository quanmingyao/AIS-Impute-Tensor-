function [ RMSE ] = AISPredictSparse( U, V, tstD, S )

if(exist('S', 'var'))
    U{1} = U{1}*S{1};
    U{2} = U{2}*S{2};
end

M = tstD.m;
N = tstD.n;
L = length(tstD.row);

nnZ = 0;
sqr = 0;

for l = 1:L
    Ut = U{1}; Vt = V{1};
    
    Um = Ut(M*(l-1) + 1: M*l,:);
    part = partXY_blas(Um', Vt', tstD.row{l}, tstD.col{l}, length(tstD.row{l}));
    
    Ut = U{2}; Vt = V{2};

    Vm = Vt(N*(l-1) + 1: N*l,:);
    part = part + partXY_blas(Ut', Vm', tstD.row{l}, tstD.col{l}, length(tstD.row{l}));
    
    nnZ = nnZ + length(tstD.row{l});
    
    sqr = sqr + sum((tstD.data{l} - part').^2);
end

RMSE = sqrt(sqr/nnZ);

end

