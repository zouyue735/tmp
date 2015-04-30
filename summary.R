rm(list=ls());

remove_zero <- FALSE;

train <- read.csv('train.csv');

levels <- list(rep(0,93));
summaries <- list(rep(0,93));
feat_levels <- list(rep(0,93));
feat_summaries <- list(rep(0,93));

for(i in 1:93) {
	train_i <- train[[paste('feat_', i, sep='')]];
	levels[[i]] <- summary(factor(train_i));
	summaries[[i]] <- summary(train_i);
	feat_levels[[i]] <- list(rep(0,9));
	feat_summaries[[i]] <- list(rep(0,9));
	for(j in 1:9) {
		train_ij <- train_i[train$target==paste('Class_', j, sep='')];
		feat_levels[[i]][[j]] <- summary(factor(train_ij));
		feat_summaries[[i]][[j]] <- summary(train_ij);
	}
}
