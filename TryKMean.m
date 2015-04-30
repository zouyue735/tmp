clear;
clc;
close all;

load('train.mat');

train(:,1)=[];
label=train(:,94);
train=train(:,1:93);

nCluster = 5 * ones(9,1);
epsilon = 0.000001;
for i = 1:9
	figure(i)
	hold on
	trainOfClass = train(label == i, :);
	for nCluster = 1:100
		[centers clusterIdxs dists] = KMeans(trainOfClass, nCluster, epsilon);
		dtotal = sum(dists) / size(trainOfClass, 1);
		plot(nCluster, dtotal, 'b*');
	end
	#axis([0 110 0 400]);
	
	#cdist = zeros(size(centers, 1), size(centers, 1));
	#for j = 1:size(centers, 1)
	#	for k = 1:size(centers, 1)
	#		cdist(j, k) = sum((centers(j,:) - centers(k,:)).^2);
	#	end
	#end
	#disp(strcat('class ', num2str(i)));
	#disp('centers');
	#disp(cdist);
	#disp('clusters');
	#[a b] = table(clusterIdxs);
end	

