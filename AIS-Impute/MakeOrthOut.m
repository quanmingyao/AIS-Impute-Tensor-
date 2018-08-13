function [U, S, V] = MakeOrthOut(U, V)

Modes = length(U);

S = cell(Modes, 1);
for mode = 1:Modes
    [Qu, Ru] = qr(U{mode}, 0);
    [Qv, Rv] = qr(V{mode}, 0);
    
    [Um, Sm, Vm] = svd(Ru*Rv', 'econ');
    
    U{mode} = Qu*Um;
    S{mode} = Sm;
    V{mode} = Qv*Vm;
end

end