function [ traD, test ] = generateYouTube( network, ratio, saps )

sz = size(network{1}, 1);

rIdx = randperm(sz);
rIdx = rIdx(1:min(sz, saps));

for i = 1:length(network)
    network{i} = network{i}(rIdx, rIdx);
end

% make up data
data = cat(2, network{1}, network{2}, network{3}, network{4}, network{5});
[~, ~, val] = find(data);
allMean = mean(val);
allStd  = std(val);

clear data;

traD = cell(length(network), 1);
for i = 1:length(network)   
    [row, col, val] = find(network{i});
    
    val = val - allMean;
    val = val/allStd;
    
    m = size(network{i}, 1);
    n = size(network{i}, 2);
    
    idx = randperm(length(val));
    
    traIdx = idx(1:floor(length(idx)*ratio));
    tstIdx = idx(floor(length(idx)*ratio) + 1 : end);
    
    traD{i} = sparse(row(traIdx), col(traIdx), val(traIdx), m, n);
    
    test.row{i}  = row(tstIdx);
    test.col{i}  = col(tstIdx);
    test.data{i} = val(tstIdx);
    test.m = m;
    test.n = n;
end

end

