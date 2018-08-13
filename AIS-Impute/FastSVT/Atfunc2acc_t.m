function [x] = Atfunc2acc_t(x, U1, V1, U0, V0, spa)

x1 = V1*(U1'*x) + V0*(U0'*x);

L = length(spa);
x2 = [];
for l = 1:L
    xl = spa{l}'*x;
    x2 = cat(1, x2, xl);
end

x = x1 + x2;

end