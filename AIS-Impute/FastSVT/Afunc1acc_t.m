function [x] = Afunc1acc_t(x, U1, V1, U0, V0, spa)

x1 = U1*(V1'*x) + U0*(V0'*x);

L = length(spa);
x2 = [];
for l = 1:L
    xl = spa{l}*x;
    x2 = cat(1, x2, xl);
end

x = x1 + x2;

end