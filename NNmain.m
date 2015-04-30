clear;
clc;
close all;

ratio=0.4;

load('train.mat');

m=size(train, 1);

train=train(randperm(m),:);

test=train(1:floor(m*ratio), :);
train=train(1 + size(test, 1):m,:);

train(:,1)=1;
label=train(:,95);
train=train(:,1:94);

train=[train,train(:,2:94).*train(:,2:94)];
maxs=max(train);

m=size(train,1);
n=size(train,2);
for i=1:n
	train(:,i)=train(:,i) / maxs(i);
end

optThetas=zeros(n,9);
costs=zeros(9,1);
options = optimset('GradObj', 'on', 'MaxIter', 10000);
for class=1:9
	theta=abs(rand(n,1));

	binary_label=(label==class);
	
	[optThetas(:,class), costs(class)] = fminunc(@(t)(cost(t, train, binary_label)), theta, options);
end

train_result=sigmoid(train * optThetas);
[value train_result_label]=max(train_result');
train_result_label=train_result_label';
train_acc=sum(train_result_label==label) / size(label,1)
train_ll=logloss(train_result,label)

if ratio~=0
	test(:,1)=1;
	test_label=test(:,95);
	test=test(:,1:94);
	test=[test,test(:,2:94).*test(:,2:94),test(:,2:94).*test(:,2:94).*test(:,2:94),test(:,2:94).*test(:,2:94).*test(:,2:94).*test(:,2:94)];

	mt=size(train,1);
	nt=size(train,2);
	for i=1:nt
		test(:,i)=test(:,i) / maxs(i);
	end

	test_result=sigmoid(test*optThetas);
	[value test_result_label]=max(test_result');
	test_result_label=test_result_label';
	test_acc=sum(test_result_label==test_label) / size(test_label,1)
	test_ll=logloss(test_result,test_label)
else
	load('test.mat');

	t(:,1)=1;
	t=[t,t(:,2:94).*t(:,2:94)];

	mt=size(train,1);
	nt=size(train,2);
	for i=1:nt
		t(:,i)=t(:,i) / maxs(i);
	end
	t_result=sigmoid(t*optThetas);
	save('result.mat', 't_result');
end
