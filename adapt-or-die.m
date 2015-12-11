%% AttentionBased-Feedback
% Felix Meissner, 11th of Dec, 2015
% felix.meissner@uzh.ch

%clear all 
%close all

%rand('state',sum(100*clock));      % set/initialize randomization
rand('state',7);                    % make sim reproducible  
rng(7,'twister');

%% Set initial parameters of Simulation
% 

replications    = 750;              % # of replications
arms            = 10;               % # of arms in bandit model
periods         = 200;              % # of periods agents learn
variations      = 9;                % # of variations
                                    % variations can be modifications of
                                    % various parameters, such as arms,
                                    % agents, learning rates, # of layers,
                                    % etc.

agents          = 7;                % # of individuals
beta            = 0.5;              % learning rate
epsilon         = 0.05;             % propensity of agent to explore (epsilon-greedy)
noise           = 0.2;              % noise-factor of feedback

slmode          = 1;                % social learning mode
delta           = 0.2;              % probability that social learning is preferred over individual learning
alpha           = 0.25;             % rate at which social learning happens

tlayers         = 4;                % total # of layers of MABs
alayers         = 2;                % # of layers attributed to each individual for fitness (out of tlayers)
%flayers         = 3;               % # of layers looked at for learning (out of ilayers)
flayersimpact   = 0.5;

pshock          = 0;%.1;                % Probability that a shock happens                          
shockimpact     = 0.5;%.5;                % How many positions (percentage) are affected?


tmpresults=zeros(periods,agents);                 % For recording results
finresults=zeros(11,variations,replications,periods,agents);   % for final results

flimpact = zeros(1,tlayers);
for ix = 1:alayers
    flimpact(ix) = 0.5^(ix);
end

%% Simulation / Replication
%

for Chet = 1:11
    pshock = (Chet-1)*0.05
    %tlayers = 1+Chet
for Cvar = 1:variations
    delta = (Cvar-1)*0.125
    %tlayers = 1+2*Cvar
    
    for Crep = 1:replications           % Counter for replications

        % ----- Create Fitness Landscape ----- %
        fl = rand(tlayers,arms);

        % ----- Attribute which Landscape matters per agent with which impact ----- %
        flmatch = zeros(agents,tlayers);
                %?!? flmpos = randperm(tlayers,alayers);
                %?!? [~, out] = sort(rand(tlayers,alayers),1);
            for rows = 1:agents
                pos = randperm(tlayers,alayers);
                flmatch(rows,pos) = flimpact(1:alayers);
            end

        % ----- Create Agent's Expectations ----- %
        beliefs = zeros(agents,arms);
        socialbeliefs = zeros(agents,arms);


        for Cper = 1:periods            % Counter for periods

            % ----- Probability that a shock/ (partly) reset happens ----- %
            if binornd(1,pshock)==1
                    whichflpos = rand(tlayers,arms) < shockimpact;      % Shock only affects some positions...
                    howflchanges = rand(tlayers,arms);                  % ... and assigns a new rand-number!

                    fl(whichflpos) = howflchanges(whichflpos);          % ... execute by overwriting the positions
            end


            % ----- Start here with selection and updating rules!  ----- %

             for Cag = randperm(agents)           % random selection which agent starts acting!

                 % ----- Determine which arm is chosen ---- % 
                 if binornd(1,epsilon)==1                               % Take an arm at random
                        choice = randi(arms);
                 elseif binornd(1,delta)==1
                     abelief = squeeze(socialbeliefs(Cag,:));
                     [~,choice]=max(abelief + rand(1,arms)/1000);       % agent decides greedy/ highest expected value
                 else
                     abelief = squeeze(beliefs(Cag,:));
                     [~,choice]=max(abelief + rand(1,arms)/1000);       % agent decides greedy/ highest expected value
                 end

                 % ----- Determine rewards & payoffs ---- %
                 whichflmatch       = squeeze(flmatch(Cag,:)');
                 subjectiveReward   = sum(fl(:,choice).*whichflmatch) + randn*noise;
                 realReward         = sum(fl(:,choice).*whichflmatch);


                 % ----- Update Beliefs ---- %
                 beliefs(Cag,choice)  = beliefs(Cag,choice)*(1-beta) + subjectiveReward*beta;


                 % ----- Learn from Others ---- %
                 % implement here...
                 % ... with certain probability
                 % ... if similar enough
                 % ... if rather distant/ less similar
                 % ... if underperforming - compared to previous period
                 % ... 

                 switch(slmode)
                     case 1
                         socialbeliefs(:,choice) = socialbeliefs(:,choice)*(1-alpha) + subjectiveReward*alpha;
                     case 2
                 end




                 % -----  record results ---- %
                 tmpresults(Cper,Cag) = realReward;
                 %finresults(Chet,Cvar,Crep,Cper,Cag) = realReward;
             end
        end
        finresults(Chet,Cvar,Crep,:,:) = tmpresults;    
    end
end
end
sumstatoverall = mean(mean(finresults,5),3);
%% Prepare for Vizualization & Analysis

% ----- Rescale ----- %
% rescale that results in range [0 1]
% 
% plot(sumstat)

%i_legend=legend('P of social learning = 0','P of social learning = 0.2','P of social learning = 0.6','P of social learning = 1');
%ii_legend=legend('P of social learning = 0','P of social learning = 0.2','P of social learning = 0.6','P of social learning = 0.8','P of social learning = 1');


sumstat = squeeze(sumstatoverall(1,:,:,:));
figure('Units','inches','Position',[0 0 6.3 9.7],'PaperPositionMode','auto');
plot(1:periods,squeeze(sumstat(1,:,:)),'color',[0.1, 0.1, 0.44],'linewidth',3);
hold all
%plot(1:periods,squeeze(sumstat(2,:,:)),'color',[1.0, 0.0, 1.0],'linewidth',3);
plot(1:periods,squeeze(sumstat(3,:,:)),'color',[0.19, 0.84, 0.78],'linewidth',3);
%plot(1:periods,squeeze(sumstat(4,:,:)),'color',[0.85,0.85,0.9],'linewidth',3);
plot(1:periods,squeeze(sumstat(5,:,:)),'color',[1.0, 0.1, 0.04],'linewidth',3);
%plot(1:periods,squeeze(sumstat(6,:,:)),'color',[0.1, 1.0, 0.44],'linewidth',3);
%plot(1:periods,squeeze(sumstat(7,:,:)),'color',[0.1, 1.0, 0.44],'linewidth',3);
plot(1:periods,squeeze(sumstat(8,:,:)),'color',[0.1, 1.0, 0.44],'linewidth',3);
ylim([0.35 0.6]);
ylabel('Performance','FontSize', 15)
xlabel('# of Periods','FontSize', 15)
%line('XData', [0 pperiods], 'YData', [0.725 0.725], 'LineStyle', ':','LineWidth', 2, 'Color',[0.85,0.85,0.9]);
%set(i_legend,'FontSize', 13, 'Location','SouthEast', 'FontSize', 13);
legend('delta = 0', 'delta = 0.2','delta = 0.6','delta = 1', 'FontSize', 17, 'Location','SouthEast') %,'FontSize', 13, 'Location','SouthEast');
hold off
print -depsc2 adaptordie0.eps


% short run


% long run


