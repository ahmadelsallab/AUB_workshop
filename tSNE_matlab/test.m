% Load data
load 'mnist_train.mat'
ind = randperm(size(train_X, 1));
train_X = train_X(ind(1:5000),:);
train_labels = train_labels(ind(1:5000));
% Set parameters
no_dims = 2;
init_dims = 30;
perplexity = 30;
% Run t-SNE
mappedX = tsne(train_X, [], no_dims, init_dims, perplexity);
% Plot results
gscatter(mappedX(:,1), mappedX(:,2), train_labels);
