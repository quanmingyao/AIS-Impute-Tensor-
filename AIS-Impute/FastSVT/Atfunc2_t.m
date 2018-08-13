function [x] = Atfunc2_t(x, U, V, spa)

x1 = V*(U'*x);

L = length(spa);
x2 = [];
for l = 1:L
    xl = spa{l}'*x;
    x2 = cat(1, x2, xl);
end

x = x1 + x2;

end