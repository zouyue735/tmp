function ll=logloss(result,label)
	m=size(result,1);
	ll=0;
	for i=1:m
		ll=ll+log(result(i,label(i)));
	end
	ll=-ll/m;
end
