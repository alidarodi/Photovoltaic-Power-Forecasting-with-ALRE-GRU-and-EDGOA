clc
close all
clear all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Load Time Series Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data, ~] = xlsread('Data_All.csv'); 
timeSeries1 = data(1050:1350)'; 
timeSeries2 = data(3000:3300)'; 

timeSeries=timeSeries1;

timeSeries_augmented = Augment_fun(timeSeries);

% Normalize the data (zero mean, unit variance)
if std(timeSeries_augmented) == 0
    errors('Standard deviation of timeSeries is zero. Cannot normalize.');
end
Mean_=mean(timeSeries_augmented);
Std_=std(timeSeries_augmented);
timeSeries_normalized = (timeSeries_augmented - mean(timeSeries_augmented)) / std(timeSeries_augmented);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Apply VMD to Decompose Time Series
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_imf = 9;  % Number of modes (IMFs)
alpha = 2000; % Moderation parameter for VMD

% VMD decomposition with corrected syntax
[imfs, u_hat, omega] = vmd(timeSeries_normalized, ...
    'NumIMFs', num_imf, ...
    'PenaltyFactor', alpha);

% Check imfs dimensions and validity
[numTimeSteps, numIMFs] = size(imfs);

figure()
subplot(5,2,1)
plot(timeSeries_augmented)
xlabel('Sample Point')
title('Original Augmented Data')
subplot(5,2,2)
plot(imfs(:,1))
xlabel('Sample Point')
title('IMF 1')
subplot(5,2,3)
plot(imfs(:,2))
xlabel('Sample Point')
title('IMF 2')
subplot(5,2,4)
plot(imfs(:,3))
xlabel('Sample Point')
title('IMF 3')
subplot(5,2,5)
plot(imfs(:,4))
xlabel('Sample Point')
title('IMF 4')
subplot(5,2,6)
plot(imfs(:,5))
xlabel('Sample Point')
title('IMF 5')
subplot(5,2,7)
plot(imfs(:,6))
xlabel('Sample Point')
title('IMF 6')
subplot(5,2,8)
plot(imfs(:,7))
xlabel('Sample Point')
title('IMF 7')
subplot(5,2,9)
plot(imfs(:,8))
xlabel('Sample Point')
title('IMF 8')
subplot(5,2,10)
plot(imfs(:,9))
xlabel('Sample Point')
title('IMF 9')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Prepare Data for GRU 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameters
sequenceLength = 10; % Number of time steps in each input sequence
numFeatures = num_imf; % Number of IMFs (features)
numResponses = 1; % Predict reconstructed time series (scalar)
trainRatio = 0.66; % Training data ratio

% Convert IMFs to sequences
numSamples = numTimeSteps - sequenceLength; % Corrected calculation
X = zeros(numFeatures, sequenceLength, numSamples);
Y = zeros(numResponses, numSamples);

for i = 1:numSamples
    X(:, :, i) = imfs(i:i+sequenceLength-1, :)'; % Input sequence (IMFs as features)
    Y(1, i) = sum(imfs(i+sequenceLength, :), 2); % Target: Sum of IMFs (scalar)
end

% Split into training and test sets
numTrain = round(trainRatio * numSamples);
XTrain = X(:, :, 1:numTrain);
YTrain = Y(:, 1:numTrain);
XTest = X(:, :, numTrain+1:end);
YTest = Y(:, numTrain+1:end);

% Convert to cell arrays for LSTM training
XTrainCell = cell(numTrain, 1);
for i = 1:numTrain
    XTrainCell{i} = double(squeeze(XTrain(:, :, i))); % Ensure double precision
end

XTestCell = cell(numSamples - numTrain, 1);
for i = 1:numSamples-numTrain
    XTestCell{i} = double(squeeze(XTest(:, :, i))); % Ensure double precision
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% EDGOA Algorithm 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inputs=XTrainCell;
targets=YTrain;
[Solutions] = EDGOA(inputs,targets,Mean_,Std_,numFeatures);
numHiddenUnits_best=round(Solutions(1));
learningRate_best=Solutions(2);
miniBatchSize_best=round(Solutions(3));
gradientThreshold_best=round(Solutions(4));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GRU Network 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numChannels = numFeatures; % Match number of features in data
numResponses = 1;          % Single output for regression
maxPosition = 256;
numHeads = 4;
numKeyChannels = numHeads * 32;
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
maxEpochs = 100;
miniBatchSize = miniBatchSize_best;  % 32;
learningRate = learningRate_best;    % 0.001;
solver = 'adam';
shuffle = 'every-epoch';
gradientThreshold = gradientThreshold_best; % 10;
executionEnvironment = 'auto';

alpha2 = 0.01;   % ضریب تأثیر گرادیان
beta2  = 0.005;  % ضریب تأثیر تغییرات loss

% --- تقسیم داده‌های آموزش به Train و Validation ---
valRatio = 0.2;   % 20% از داده‌های آموزش برای اعتبارسنجی
numVal   = round(valRatio * numTrain);
numTrainFinal = numTrain - numVal;

XTrainFinal = XTrainCell(1:numTrainFinal);
YTrainFinal = YTrain(:, 1:numTrainFinal)';

XVal = XTrainCell(numTrainFinal+1:end);
YVal = YTrain(:, numTrainFinal+1:end)';

% --- تنظیمات آموزش با لاگ کامل ---
options = trainingOptions(solver, ...
    'MaxEpochs', maxEpochs, ...
    'MiniBatchSize', miniBatchSize, ...
    'InitialLearnRate', learningRate, ...
    'Shuffle', shuffle, ...
    'GradientThreshold', gradientThreshold, ...
    'ExecutionEnvironment', executionEnvironment, ...
    'ValidationData', {XVal, YVal}, ...          % اضافه شد
    'ValidationFrequency', 10, ...               % هر 10 iteration یکبار validation
    'OutputFcn', @(info) adaptiveBetaAndLR_Fcn(info, alpha2, beta2), ...
    'Plots', 'training-progress', ...
    'Verbose', 1, ...                            % چاپ در Command Window
    'VerboseFrequency', 1);                      % هر epoch چاپ شود


% شروع ذخیره‌سازی خروجی Command Window
clc
diary('Training_Logs.txt');   % نام فایل را می‌توانید تغییر دهید
diary on;

%%%%% Train Network  
[net, info] = trainNetwork(XTrainFinal, YTrainFinal, layers, options);

diary off;   % توقف ذخیره‌سازی


%%%% Predict on Training and Test Sets
YPredTrain = predict(net, XTrainCell, 'MiniBatchSize', miniBatchSize);
YPredTest = predict(net, XTestCell, 'MiniBatchSize', miniBatchSize);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Evaluation of Results
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reconstruct time series from predicted values
trainPred = YPredTrain; % Already scalar
testPred = YPredTest;
trainTrue = YTrain'; % Already scalar
testTrue = YTest';

% Denormalize predictions (assuming original mean and std)
trainPred = trainPred * std(timeSeries_augmented) + mean(timeSeries_augmented);
testPred = testPred * std(timeSeries_augmented) + mean(timeSeries_augmented);
trainTrue = trainTrue * std(timeSeries_augmented) + mean(timeSeries_augmented);
testTrue = testTrue * std(timeSeries_augmented) + mean(timeSeries_augmented);

%%% Visualize Results
% Training data
figure;
plot(trainTrue(1:300), 'b', 'LineWidth', 2);
hold on;
plot(trainPred(1:300), '--r', 'LineWidth', 2);
legend('True', 'Predicted');
title('Training Data: True vs Predicted Time Series');
xlabel('Time Step');
ylabel('Value');
grid on;

% Test data
figure;
plot(testTrue(1:300), 'b', 'LineWidth', 2);
hold on;
plot(testPred(1:300), '--r', 'LineWidth', 2);
legend('True', 'Predicted');
title('Test Data: True vs Predicted Time Series');
xlabel('Time Step');
ylabel('Value');
grid on;

%%% Compute Metrics
testTrue_normalized=rescale(testTrue,0,1);
testPred_normalized=rescale(testPred,0,1);

MAE=mae(testTrue,testPred)
RMSE=rmse(testTrue,testPred)
MAPE=100*(sum(sum((abs(testTrue-testPred))/testTrue)))/length(testTrue)

SS_res = sum((testTrue - testPred).^2);
SS_tot = sum((testTrue - mean(testTrue)).^2);
R2 = 1 - (SS_res / SS_tot)


% Comparing predicted VS true values
figure;
subplot(3,1,1)
plot(testTrue(1:300), 'b', 'LineWidth', 2);
xlabel('Time (hours)');
ylabel('PV Power (KW)');
title('Real Power using Test Data');
subplot(3,1,2)
plot(testPred(1:300), 'r', 'LineWidth', 2);
xlabel('Time (hours)');
ylabel('PV Power (KW)');
title('Predicted Power using Test Data');
subplot(3,1,3)
errors=(testTrue_normalized-testPred_normalized);
plot(errors(1:300), 'g', 'LineWidth', 2);
xlabel('Time (hours)');
ylabel('PV Power (kW)');
title('Normalized error of Power Prediction using Test Data');
ylim([-.1 .1])

% Plot Regression
figure()
plotregression(testPred,testTrue)





