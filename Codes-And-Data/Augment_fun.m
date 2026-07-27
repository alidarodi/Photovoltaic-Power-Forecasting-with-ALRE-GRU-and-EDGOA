function [features_augmented] = Augment_fun(features)

alpha=0.01; % The coefficient of summative feature influence
gama=0.01;  % Noise Power Factor


num_rep=3;
features_rep=repmat(features,num_rep-1,1);
new_feature=features_rep+alpha*features_rep;    % Creating a new feature by linear addition
                                                % with a coefficient of the previous features
features_new=new_feature+(gama*randn(1,size(features_rep,2)));   % Add random value (noise) in new feature
features_augmented=[features;features_new];


end