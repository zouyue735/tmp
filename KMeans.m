function [centers clusterIdxs distances] = KMeans(data, nCluster, epsilon)
    nPoint = size(data, 1);
    centers = data(randperm(nPoint)(1:nCluster), :);
    distances = zeros(nCluster, 1);
    clusterIdxs = zeros(nPoint, 1);
    
    devi = epsilon;
    while devi >= epsilon
        clusterIdxs = assignClusterIdx(data, centers);
        [newCenters distances] = moveCenters(data, clusterIdxs);
        if size(newCenters, 1) == size(centers, 1)
            devi = sum((newCenters - centers).^2);
        end
        centers = newCenters;
    end
end

function clusterIdxs = assignClusterIdx(data, centers)
    nPoint = size(data, 1);
    nCluster = size(centers, 1);
    clusterIdxs = zeros(nPoint, 1);
    for i = 1:nPoint
        distMin = Inf;
        clusterIdx = -1;
        for j = 1:nCluster
            dist = sum((centers(j, :) - data(i, :)).^2);
            if dist < distMin
                distMin = dist;
                clusterIdx = j;
            end
        end
        clusterIdxs(i, 1) = clusterIdx;
    end
end

function [centers distances] = moveCenters(data, clusterIdxs)
    nPoint = size(data, 1);
    uniqueClusterIdxs = unique(clusterIdxs')';
    nCluster = size(uniqueClusterIdxs, 1);
    centers = zeros(nCluster, size(data, 2));
    distances = zeros(nCluster, 1);
    for i = 1:nCluster
        dataInCluster = data(clusterIdxs == uniqueClusterIdxs(i, 1), :);
        if size(dataInCluster, 1) == 1
            centers(i,:) = dataInCluster;
            distances(i, 1) = 0;
        else
            centers(i,:) = mean(dataInCluster);
            distances(i,1) = sum(var(dataInCluster) * size(dataInCluster, 1));
        end
    end
end
