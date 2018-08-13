function [x] = Afunc2_t(x, U, V, spa)

x1 = U*(V'*x);

L = length(spa);
[~, N] = size(spa{1});

x2 = zeros(size(x1));
for l = 1:L
    xl = x((l-1)*N + 1: l*N, :);
    x2 = x2 + spa{l}*xl;
end

x = x1 + x2;

end