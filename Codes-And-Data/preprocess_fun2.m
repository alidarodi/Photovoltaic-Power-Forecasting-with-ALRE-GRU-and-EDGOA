function [timeSeries_preprocessed_tr,timeSeries_preprocessed_ts] = preprocess_fun2(DataStore)

% ==============================================
% ====== Chack NaN and Missing Values
% ==============================================

% Get all field names in DataStore
fields = fieldnames(DataStore);
for i = 1:numel(fields)

    % Read current field
    Data = DataStore.(fields{i});

    % Process only numeric arrays
    if ~isnumeric(Data)
        continue;
    end

    % Step 1: Detect and replace NaN values
    if any(isnan(Data(:)))
        Data = fillmissing(Data,'linear',1);
    end

    % Step 2: Detect and replace missing values
    if any(ismissing(Data(:)))
        Data = fillmissing(Data,'linear',1);

        % if missing data was at the start or the end of timeseries
        if any(ismissing(Data(:)))
            Data = fillmissing(Data,'nearest',1);
        end
    end

    % Update DataStore
    DataStore.(fields{i}) = Data;

end


% ==============================================
% ====== Apply Data Normalization if Needed
% ==============================================

% % Get all field names in DataStore
% fields = fieldnames(DataStore);
% 
% for i = 1:numel(fields)
% 
%     % Read current field
%     Data = DataStore.(fields{i});
% 
%     % Process only numeric arrays
%     if ~isnumeric(Data)
%         continue;
%     end
% 
%     % Normalize data to [0,1]
%     DataMin = min(Data,[],1);
%     DataMax = max(Data,[],1);
% 
%     Data = (Data - DataMin) ./ (DataMax - DataMin);
% 
%     % Update DataStore
%     DataStore.(fields{i}) = Data;
% 
% 
% end


% ==============================================
% ====== Applying data scaling and adding noise 
% ====== to the data to improve model performance 
% ====== when faced with data that has a new scale or contains noise.
% ====== This process makes the model resistant against overfitting and also enhance model and evaluation scalibility 
% ==============================================


alpha=0.001; % The coefficient of summative feature influence
gama=0.001;  % Noise Power Factor

num_rep=3;
timeseries_rep=repmat(DataStore.timeseries,num_rep,1);

% ====== Create Final preprocessed Train Data
timeseries_rep_tr=timeseries_rep(1:num_rep*size(DataStore.timeseries_tr,1));
timeseries_tr_scaled=timeseries_rep_tr+alpha*timeseries_rep_tr;                       % Creating a new feature by linear addition
                                                                                      % with a coefficient of the previous features
timeseries_tr_noisy=timeseries_tr_scaled+(gama*randn(1,size(timeseries_rep_tr,2)));   % Add random value (noise) in new feature
timeSeries_preprocessed_tr=timeseries_tr_noisy;


% ====== Test Data
timeseries_rep_ts=timeseries_rep(end-(num_rep*size(DataStore.timeseries_ts,1))+1:end);
timeseries_ts_scaled=timeseries_rep_ts+alpha*timeseries_rep_ts;                       
timeseries_ts_noisy=timeseries_ts_scaled+(gama*randn(1,size(timeseries_rep_ts,2)));   
timeSeries_preprocessed_ts=timeseries_ts_noisy;



end