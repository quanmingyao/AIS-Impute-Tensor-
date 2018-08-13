function [x] = Atfunc1_t(x, U, V, spa)

x1 = V*(U'*x);

L = length(spa);
[M, ~] = size(spa{1});

x2 = zeros(size(x1));
for l = 1:L
    xl = x((l-1)*M + 1: l*M, :);
    x2 = x2 + spa{l}'*xl;
end

x = x1 + x2;

end