% Load data
clear, clc;
load 'input_data_auto_encoder_codes_20.mat';
train_X = mTrainFeatures;
[z, train_labels] = max(mTrainTargets, [], 2);
ind = randperm(size(train_X, 1));
train_X = train_X(ind(1:end),:);
train_labels = train_labels(ind(1:end));
% Set parameters
no_dims = 2;
init_dims = 30;
perplexity = 30;
% Run t-SNE
mappedX = tsne(train_X, [], no_dims, init_dims, perplexity);
% Plot results
gscatter(mappedX(:,1), mappedX(:,2), train_labels);
