function [cost gradient]=forwardbackwardPropagation(data,label,thetas)
    nLayers = size(thetas,1);
    m = size(data,1);
    n = size(data,2);
    a = cell(nLayers + 1,1);
    d = cell(nLayers + 1,1);
    a{1,1} = data';
    for i = 1:nLayers
        a{i,1} = [ones(1,m);*a{i,1}];
        a{i + 1,1} = sigmoid(thetas{i,1}*a{i,1});
    end
    d{nLayers + 1,1} = a{nLayers + 1,1}-label;
    for i = 1:nLayers
        d{nLayers + 1 - i,1} = theta{nLayers + 1 - i,1}' * d{nLayers + 2 - i,1} .* a{nLayers + 1 - i,1} .* (1 - a{nLayers + 1 - i,1});
    end
    cost = sum(sum(- label .* log(a{nLayers + 1,1}) - (1 - label) .* log(1 - a{nLayers + 1,1}))) / m;
    gradient = cell(nLayers,1);
    for l = 1:nLayers
        for j = 1:size(a{l,1},1)
            for i = 1:size(d{l+1,1},1)
                gradient{l,1}(i,j) = mean(a{l,1}(j,:) .* d{l+1,1}(i,:));
            end
        end
    end
end

function [cost gradient]=NeuralNetworkCost(data,label,layers,thetas)
    l=size(layers,1);
    n=size(data,2);
    layers=[n;layers];
    thetaCells = cell(l,1);
    numbers=zeros(l,1);
    for i=1:l
        start=sum(numbers);
        numbers(i,1)=layers(i)*layers(i+1);
        thetaCells{i,1}=reshape(thetas((1+start):sum(numbers),1),layers(i+1),layers(i));
    end
    [cost gradientCells]=forwardbackwardPropagation(data,label,thetaCells);
    gradient=[];    
    for i=1:l
        gradient=[gradient;gradientCells{i,1}(:)];
    end
end
