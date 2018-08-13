function [x] = Afunc2acc_t(x, U1, V1, U0, V0, spa)

x1 = U1*(V1'*x) + U0*(V0'*x);

L = length(spa);
[~, N] = size(spa{1});

x2 = zeros(size(x1));
for l = 1:L
    xl = x((l-1)*N + 1: l*N, :);
    x2 = x2 + spa{l}*xl;
end

x = x1 + x2;

end