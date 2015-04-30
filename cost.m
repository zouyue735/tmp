function [c grad]=cost(theta, x, y)
	n = size(theta,1);
	z = x * theta;
	h = sigmoid(z);
	c = mean(- y .* log(h) - (1 - y) .* log(1 - h));
	grad = zeros(n,1);
	for j = 1:n
		grad(j) = mean((h - y) .* x(:,j));
	end
end
