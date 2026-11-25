clc
close all
clear all

%%%%%% For Run on GPU
parallel.gpu.enableCUDAForwardCompatibility(true) % For Install GPu for first Run
% gpuDevice(1)   % انتخاب اولین کارت گرافیک
ExecutionEnvironment = "gpu";

%%%%%%%%%%%% For PLot Figures on One Tab or Several Tabs
set(0,'DefaultFigureWindowStyle','normal')
% set(0,'DefaultFigureWindowStyle','docked')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Load Time Series Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[data, ~] = xlsread('Data_Prediction_60_Hours_BL.csv'); 
timeSeries = data(:, 2); % Assuming univariate time series (single column)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Split ORIGINAL time series before VMD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainRatio = 0.66;
windowLength = 5;
numModes = 9;

N = numel(timeSeries);
nTrain = floor(trainRatio*N);

timeSeries_tr_=timeSeries(1:nTrain);
timeSeries_ts_=timeSeries(nTrain+1:end);


DataStore = struct();
DataStore.timeseries=[timeSeries_tr_;timeSeries_ts_];
DataStore.timeseries_tr=timeSeries_tr_;
DataStore.timeseries_ts=timeSeries_ts_;

[timeSeries_preprocessed_tr,timeSeries_preprocessed_ts] = preprocess_fun2(DataStore);

trainSeries=timeSeries_preprocessed_tr;
testSeries=timeSeries_preprocessed_ts;


% Normalize using TRAINING statistics only.
if any(~isfinite(trainSeries)) || any(~isfinite(testSeries))
    error('Input contains NaN or Inf.');
end
contextLength = max(64,4*windowLength);

mu = mean(trainSeries);
sigma = std(trainSeries);
if ~isfinite(sigma) || sigma <= 0
    error('Training standard deviation must be finite and positive.');
end
xTrain = (trainSeries-mu)/sigma;
xTest = (testSeries -mu)/sigma;
x = [xTrain; xTest];

nTrain_augmented=size(xTrain,1);
N_tot=size(x,1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Causal rolling VMD: independent train/test series
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Causal VMD on training history...\n');
[XTrainCell,YTrain] = makeVMDWindowsCausal_fun(xTrain,numModes,windowLength);
fprintf('Causal VMD on test history...\n');
[XTestCell,YTest] = makeVMDWindowsCausal_fun(xTest,numModes,windowLength);

YTrain=YTrain';
YTest=YTest';
numFeatures = numModes;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% EDGOA Algorithm 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inputs=XTrainCell;
targets=YTrain;
[Solutions] = EDGOA(inputs,targets,mu,sigma,numFeatures);
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
numTrain=size(XTrainCell,1);
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
% clc
% diary('Training_Logs.txt');   % نام فایل را می‌توانید تغییر دهید
% diary on;

%%%%% Train Network  
[net, info] = trainNetwork(XTrainFinal, YTrainFinal, layers, options);

% diary off;   % توقف ذخیره‌سازی
 

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
trainPred = trainPred * sigma + mu;
testPred = testPred * sigma + mu;
trainTrue = trainTrue * sigma + mu;
testTrue = testTrue * sigma + mu;

%%% Visualize Results
% Test data
figure;
plot(data(:,1)/10,testTrue(end-size(data(:,1),1)+1:end), 'b', 'LineWidth', 2);
hold on;
plot(data(:,1)/10,testPred(end-size(data(:,1),1)+1:end), '--r', 'LineWidth', 2);
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
plot(data(:,1)/10,testTrue(end-size(data(:,1),1)+1:end), 'b', 'LineWidth', 2);
xlabel('Time (hours)');
ylabel('PV Power (KW)');
title('Real Power using Test Data');
subplot(3,1,2)
plot(data(:,1)/10,testPred(end-size(data(:,1),1)+1:end), 'r', 'LineWidth', 2);
xlabel('Time (hours)');
ylabel('PV Power (KW)');
title('Predicted Power using Test Data');
subplot(3,1,3)
error_ts_1month=(testTrue_normalized-testPred_normalized);
plot(data(:,1)/10,error_ts_1month(end-size(data(:,1),1)+1:end), 'g', 'LineWidth', 2);
xlabel('Time (hours)');
ylabel('PV Power (kW)');
title('Normalized error of Power Prediction using Test Data');
ylim([-.1 .1])

% Plot Regression
figure()
plotregression(testPred,testTrue)



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Block-Based Monte Carlo Robustness Evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n============================================\n');
fprintf('Block-Based Monte Carlo Robustness Test\n');
fprintf('============================================\n');

% Experiment Settings
numBlocks = 5;
numRuns = 10;

noiseLevel = 0.03;  % Noise STD = 3% 
noiseMean = 0;

% Noise standard deviation in normalized domain
noiseStd = noiseLevel;

% Fixed random seed for reproducibility
rng(42,'twister');

% Test Data
xTestOriginal = double(xTest(:));
nTest = numel(xTestOriginal);

if nTest < numBlocks
    error('Number of test samples must be >= numBlocks.');
end

% Original targets in physical units
YTrue = xTestOriginal*sigma + mu;

% Divide Test Data into Consecutive Blocks
blockEdges = floor((0:numBlocks)*nTest/numBlocks);

% Store Monte Carlo evaluation metrics
RMSE_MC = zeros(numBlocks,numRuns);
MAE_MC  = zeros(numBlocks,numRuns);
MAPE_MC = zeros(numBlocks,numRuns);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Monte Carlo Evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for b = 1:numBlocks

    idxStart = blockEdges(b)+1;
    idxEnd = blockEdges(b+1);
    idx = idxStart:idxEnd;

    fprintf('\nBlock %d/%d: Samples %d to %d\n', ...
        b,numBlocks,idxStart,idxEnd);

    % Original targets of this block
    trueBlock = YTrue(idx);

    for r = 1:numRuns

        % Independent Gaussian noise realization
        noise = noiseMean + ...
            noiseStd*randn(nTest,1);

        % Add noise to normalized test inputs
        xTestNoisy = xTestOriginal + noise;

        % Recompute causal VMD features
        [XTestNoisy,~] = makeVMDWindowsCausal_fun( ...
            xTestNoisy,numModes,windowLength);

        % Select current block
        XBlockNoisy = XTestNoisy(idx);

        % Predict using the fixed trained network
        YPredNoisy = predict(net,XBlockNoisy, ...
            'MiniBatchSize',miniBatchSize);

        % Denormalize predictions
        predBlock = double(YPredNoisy(:))*sigma + mu;

        % Prediction errors
        errors = trueBlock - predBlock;

        % RMSE
        RMSE_MC(b,r) = sqrt(mean(errors.^2));

        % MAE
        MAE_MC(b,r) = mean(abs(errors));

        % MAPE (%)
        validIdx = abs(trueBlock) > 1e-8;

        if any(validIdx)
            MAPE_MC(b,r) = 100*mean( ...
                abs(errors(validIdx)./trueBlock(validIdx)));
        else
            MAPE_MC(b,r) = NaN;
        end

        fprintf(['Run %2d/%2d | RMSE = %.6f | ', ...
                 'MAE = %.6f | MAPE = %.4f%%\n'], ...
                 r,numRuns,RMSE_MC(b,r), ...
                 MAE_MC(b,r),MAPE_MC(b,r));

    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Statistical Analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% RMSE statistics
Mean_RMSE = mean(RMSE_MC,2);
Std_RMSE = std(RMSE_MC,0,2);

% MAE statistics
Mean_MAE = mean(MAE_MC,2);
Std_MAE = std(MAE_MC,0,2);

% MAPE statistics
Mean_MAPE = mean(MAPE_MC,2,'omitnan');
Std_MAPE = std(MAPE_MC,0,2,'omitnan');

Block = (1:numBlocks)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Summary Results Table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ResultsTable = table( ...
    string(Block), ...
    Mean_RMSE,Std_RMSE, ...
    Mean_MAE,Std_MAE, ...
    Mean_MAPE,Std_MAPE, ...
    'VariableNames', ...
    {'Block','Mean_RMSE','Std_RMSE', ...
     'Mean_MAE','Std_MAE', ...
     'Mean_MAPE','Std_MAPE'});

% Average across the five blocks
AverageRow = table( ...
    "Average", ...
    mean(Mean_RMSE),mean(Std_RMSE), ...
    mean(Mean_MAE),mean(Std_MAE), ...
    mean(Mean_MAPE,'omitnan'), ...
    mean(Std_MAPE,'omitnan'), ...
    'VariableNames',ResultsTable.Properties.VariableNames);

% Append average row
ResultsTable = [ResultsTable; AverageRow];

disp(' ');
disp('========== Monte Carlo Statistical Results ==========');
disp(ResultsTable);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Prepare Data for Plotting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MetricNames = {'RMSE','MAE','MAPE (%)'};

MetricData = {RMSE_MC,MAE_MC,MAPE_MC};

MetricMeans = {Mean_RMSE,Mean_MAE,Mean_MAPE};

MetricStds = {Std_RMSE,Std_MAE,Std_MAPE};

MetricColors = [ ...
    0.00 0.35 0.75; ...
    0.85 0.33 0.10; ...
    0.20 0.60 0.30];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 1: Mean Metrics with Error Bars
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Color','w');

for k = 1:3

    subplot(3,1,k);

    errorbar(Block,MetricMeans{k},MetricStds{k}, ...
        '-o', ...
        'Color',MetricColors(k,:), ...
        'LineWidth',1.8, ...
        'MarkerSize',7, ...
        'CapSize',12);

    xlabel('Test Data Block');
    ylabel(MetricNames{k});

    title(['Monte Carlo Mean ',MetricNames{k}, ...
        ' with Standard Deviation']);

    legend('Mean \pm STD','Location','best');

    xticks(Block);
    xlim([1 numBlocks]);

    grid on;
    box on;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 2: Metric Distributions (Boxplots)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Color','w');

for k = 1:3

    subplot(3,1,k);

    boxplot(MetricData{k}', ...
        'Labels',cellstr(compose('Block %d',Block)));

    xlabel('Test Data Block');
    ylabel(MetricNames{k});

    title(['Monte Carlo ',MetricNames{k}, ...
        ' Distribution']);

    grid on;
    box on;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Figure 3: Mean Metrics with Shaded STD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure('Color','w');

for k = 1:3

    subplot(3,1,k);
    hold on;

    % Mean and standard deviation
    MeanValue = MetricMeans{k};
    StdValue = MetricStds{k};

    % Upper and lower bounds
    UpperBound = MeanValue + StdValue;
    LowerBound = MeanValue - StdValue;

    % Shaded standard deviation region
    fill([Block; flipud(Block)], ...
         [UpperBound; flipud(LowerBound)], ...
         MetricColors(k,:), ...
         'FaceAlpha',0.20, ...
         'EdgeColor','none', ...
         'DisplayName','Mean \pm STD');

    % Central mean line
    plot(Block,MeanValue,'-o', ...
        'Color',MetricColors(k,:), ...
        'LineWidth',2.2, ...
        'MarkerSize',7, ...
        'MarkerFaceColor',MetricColors(k,:), ...
        'DisplayName','Mean');

    xlabel('Test Data Block');
    ylabel(MetricNames{k});

    title(['Monte Carlo ',MetricNames{k}, ...
        ' with Shaded Standard Deviation']);

    legend('Location','best');

    xticks(1:numBlocks);
    xlim([1 numBlocks]);

    grid on;
    box on;
    set(gca,'FontSize',11);

    hold off;

end

fprintf('\nMonte Carlo Robustness Evaluation Completed.\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Export Monte Carlo Statistical Results to Excel
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fileName = 'Statistical_Results_Proposed_60H.xlsx';

% Remove previous file
if isfile(fileName)
    delete(fileName);
end

% Metrics data
MetricData = {RMSE_MC, MAE_MC, MAPE_MC};
MetricNames = {'RMSE','MAE','MAPE'};

% Excel output cell array
ExcelData = {};

rowStart = 1;

for k = 1:3

    % Metric title
    ExcelData{rowStart,1} = MetricNames{k};

    % Block headers
    for b = 1:numBlocks
        ExcelData{rowStart+1,b+1} = ...
            sprintf('Block %d',b);
    end

    % Average column header
    ExcelData{rowStart+1,numBlocks+2} = 'Average';

    % Monte Carlo results (10 runs)
    for r = 1:numRuns

        ExcelData{rowStart+1+r,1} = ...
            sprintf('Run %d',r);

        for b = 1:numBlocks
            ExcelData{rowStart+1+r,b+1} = ...
                MetricData{k}(b,r);
        end

        % Average across all blocks for this run
        ExcelData{rowStart+1+r,numBlocks+2} = ...
            mean(MetricData{k}(:,r),'omitnan');

    end

    % Mean of 10 runs for each block
    ExcelData{rowStart+numRuns+2,1} = 'Mean';

    % Standard deviation of 10 runs
    ExcelData{rowStart+numRuns+3,1} = 'STD';

    MeanValues = zeros(1,numBlocks);
    StdValues = zeros(1,numBlocks);

    for b = 1:numBlocks

        values = MetricData{k}(b,:);

        MeanValues(b) = mean(values,'omitnan');
        StdValues(b) = std(values,0,'omitnan');

        ExcelData{rowStart+numRuns+2,b+1} = MeanValues(b);
        ExcelData{rowStart+numRuns+3,b+1} = StdValues(b);

    end

    % Average of block means
    ExcelData{rowStart+numRuns+2,numBlocks+2} = ...
        mean(MeanValues,'omitnan');

    % Average of block standard deviations
    ExcelData{rowStart+numRuns+3,numBlocks+2} = ...
        mean(StdValues,'omitnan');

    % Two empty rows between metrics
    rowStart = rowStart + numRuns + 6;

end

% Write all results into one Excel sheet
% writecell(ExcelData,fileName,'Sheet','Monte Carlo Results');

fprintf('\nStatistical results saved to: %s\n',fileName);



%%% Local functions
function [X,Y] = makeVMDWindowsCausal_fun(x,K,L)
    x = double(x(:));
    nSamples = numel(x);
    if nSamples == 0
        error('Input time series must not be empty.');
    end
    % Pad with the first observation, not with zeros.
    xPadded = [repmat(x(1),L-1,1); x];
    X = cell(nSamples,1);
    Y = zeros(nSamples,1);
    for i = 1:nSamples
        % Exactly L past values; the current target is excluded.
        past = xPadded(i:i+L-1);
        pastModes = vmd(past,'NumIMFs',K,'Display',false);
        X{i} = pastModes.';
        Y(i) = x(i);
        if mod(i,100)==0 || i==nSamples
            fprintf('Causal VMD: %d / %d windows completed\n',i,nSamples);
        end
    end
end







