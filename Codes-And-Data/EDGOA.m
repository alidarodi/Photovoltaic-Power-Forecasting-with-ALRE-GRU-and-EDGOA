function [ind_best] = EDGOA(inputs,targets,Mean_,Std_,numFeatures)

%%% Problem Definition
CostFunction = @(x)Cost_Function(x,inputs,targets,Mean_,Std_,numFeatures);


nVar = 4;    % Number of Decision Variable

VarMin = [300 0.0009 125 9];    %100; % Lower Bound of Variable (Hidden Layer Size, Learning Rate, Minibatch Size, Gradian Threshold)
VarMax = [400 0.0011 128 11];    %200; % Upper Bound of Variable (Hidden Layer Size, Learning Rate, Minibatch Size, Gradian Threshold)

%%% EEDGOA Parameters
MaxIt = 30;     % Maximum Nomber of Iterations
nPop = 10;         % Population Size

% Constriction Coefeicient
cMax = 1;
cMin = 1e-4;

Params.lb = VarMin;
Params.ub = VarMax;
Params.nPop = nPop;
%%% Initialization
empty_GrassHopper.Position = [];
empty_GrassHopper.Cost = [];

GrassHopper = repmat(empty_GrassHopper,nPop,1);

for i = 1:nPop
    % Initialize Position
    GrassHopper(i).Position = unifrnd(VarMin,VarMax,[1,nVar]);
    
    
    % Evaluate
    GrassHopper(i).Cost = CostFunction(GrassHopper(i).Position);
end
NFE = nPop;
[~,indx] = min([GrassHopper.Cost]);

TargetGH = GrassHopper(indx);   % Target GrassHopper

BestCost = zeros(1,MaxIt);
MeanCost = zeros(1,MaxIt);
WorstCost = zeros(1,MaxIt);
nfe = zeros(1,MaxIt);

%%%%%%%%%%%%%%%%%%%%%%%%%% EDGOA Main Loop
%%%%%%%%%%%%% In the loop position x dynamicly update based on inertia motion weight (C_EDGOA Parameter)  
b = 1;
gama=2;
p_mut_init=0.2;
for it = 1:MaxIt
    it_EDGOA=it
    
    
    
    
    %====================================================================================================
    %============= Adjust c parameter of EDGOA Algorithm
    %====================================================================================================
    
    %%%% Old c 
    % c = cMax - it*((cMax - cMin)/MaxIt); 
    
    %%%% New c
    c = b * cMax * ( (cMin/cMax)^((it/MaxIt)^gama) );   % Update inertia motion weight (C_EDGOA Parameter) 

    %%%% Set c Bounds
    % if c2 < cMin
    %     c2=cMin;
    % elseif c2> cMax
    %     c2=cMax;
    % end
    % c3=c2

    Params.c = c;
    for i = 1:nPop
        Params.i = i;
        GrassHoppers = cat(1,GrassHopper.Position);
        SI = SocialInteraction(GrassHoppers,Params);
        GrassHopper(i).Position = c*SI + TargetGH.Position;     % GrassHopper Position = x in Equations 
                                                                % that updates dynamicly by inertia motion weight
                
        GrassHopper(i).Position = min(max(GrassHopper(i).Position,VarMin),VarMax);      % if position be out of bound then return that into 
                                                                                        % the specified bound between VanMin and VarMax
        
        % Evaluation
        GrassHopper(i).Cost = CostFunction(GrassHopper(i).Position);
        
        % Update Target
        if GrassHopper(i).Cost<TargetGH.Cost
            TargetGH = GrassHopper(i);
        end
    end

    %====================================================================================================
    %============= Apply mutation in EDGOA Algorithm
    %====================================================================================================
    r=randn;
    p_mut= p_mut_init   + ( 0.05* (it/MaxIt) );
    if p_mut<r
        nums = randperm(nPop, 3);
        ind_mutation = randperm(nPop, 1);
        p_prim= abs(GrassHopper(nums(1)).Cost)+abs(GrassHopper(nums(2)).Cost)+abs(GrassHopper(nums(3)).Cost);
        p1=abs(GrassHopper(nums(1)).Cost)/p_prim;
        p2=abs(GrassHopper(nums(2)).Cost)/p_prim;
        p3=abs(GrassHopper(nums(3)).Cost)/p_prim;
        
        GrassHopper(ind_mutation).Position= (GrassHopper(nums(1)).Position+GrassHopper(nums(2)).Position+GrassHopper(nums(3)).Position)/3 + ...
            ( (p2-p1)* (GrassHopper(nums(1)).Position-GrassHopper(nums(2)).Position) + ...
              (p3-p2)* (GrassHopper(nums(2)).Position-GrassHopper(nums(3)).Position) + ...
              (p1-p3)* (GrassHopper(nums(3)).Position-GrassHopper(nums(1)).Position)  );

        GrassHopper(ind_mutation).Position = min(max(GrassHopper(ind_mutation).Position,VarMin),VarMax);   % if position be out of bound then return that into 
                                                                                                           % the specified bound between VanMin and VarMax
    end

    BestCost(it) = TargetGH.Cost;
    MeanCost(it) = mean([GrassHopper.Cost]);
    WorstCost(it) = max([GrassHopper.Cost]);
    NFE = NFE + nPop;
    nfe(it) = NFE;
    
%     disp(['Iteration: ',num2str(it),' , NFE = ',num2str(nfe(it)),' Best Cost = ',num2str(BestCost(it))]);
     
end


ind_best = TargetGH.Position;


end



