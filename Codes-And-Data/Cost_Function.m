function z = Cost_Function(x,inputs,targets,Mean_,Std_,numFeatures)

XTrainCell=inputs;
YTrain=targets;

numHiddenUnits_best=round(x(1));
learningRate_best=x(2);
miniBatchSize_best=round(x(3));
gradientThreshold_best=round(x(4));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GRU Network 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numResponses = 1;          % Single output for regression
numHiddenUnits = numHiddenUnits_best; % 100 Number of hidden units in LSTM layers
dropoutRate = 0.2; % Dropout rate for regularization

layers = [ ...
    sequenceInputLayer(numFeatures, 'Name', 'input')
    gruLayer(numHiddenUnits, 'Name', 'lstm1')
    dropoutLayer(dropoutRate, 'Name', 'dropout1')
    gruLayer(round(numHiddenUnits/2), 'Name', 'lstm2')
    dropoutLayer(dropoutRate, 'Name', 'dropout2')
    indexing1dLayer("last")
    fullyConnectedLayer(numResponses, 'Name', 'fc')
    regressionLayer('Name', 'output')];

%%% Training Options
maxEpochs = 25;  %100
miniBatchSize = miniBatchSize_best; %32;
learningRate = learningRate_best; %0.001;
solver = 'adam';
shuffle = 'every-epoch';
gradientThreshold = gradientThreshold_best; % 10;
executionEnvironment = 'auto';


alpha2 = 0.01;   % ضریب تأثیر گرادیان
beta2  = 0.005;  % ضریب تأثیر تغییرات loss

options = trainingOptions(solver, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', learningRate, ...
    'Shuffle', shuffle, ...
    'GradientThreshold', gradientThreshold, ...
    'ExecutionEnvironment', executionEnvironment, ...
    'OutputFcn', @(info) adaptiveBetaAndLR_Fcn(info, alpha2, beta2), ...
    'Verbose', 0); %'Plots', 'training-progress', ...


%%%%% Train Network  
net = trainNetwork(XTrainCell, YTrain', layers, options);

%%%% Predict on Training and Test Sets
YPredTrain = predict(net, XTrainCell, 'MiniBatchSize', miniBatchSize);

trainPred = YPredTrain; % Already 
trainPred = trainPred * Std_ + Mean_;
trainTrue = YTrain';

RMSE=rmse(trainTrue,trainPred);
z=RMSE;


end