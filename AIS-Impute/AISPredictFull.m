function [ X ] = AISPredictFull( U, S, V, L )

M = size(U{1}, 1)/L;
N = size(V{2}, 1)/L;

X = zeros(M, N, L);
for l = 1:L
    Um = U{1}(M*(l-1) + 1: M*l,:);
    Vm = V{2}(N*(l-1) + 1: N*l,:);
    
    X(:,:,l) = X(:,:,l) + Um*S{1}*V{1}' + U{2}*S{2}*Vm';
end

end

