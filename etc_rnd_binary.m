function [ A, T ] = etc_rnd_binary(dt, E, G, nA, x0, B, lambda, xi, C, origin )
% % generates cultural traits and artifacts from Evoked and Transmitted
% Culture (ETC) model with binary traits and binary artifacts
%
%[ A, T ] = etc_rnd_binary(dt, E, G, nA, x0, b, lambda, xi, C, origin )
%
% - dt: time step (scalar)
% - E: ecological variable in each region and each time step (R x nT matrix,
% where R is the number of regions and nT the number of time steps). Use
% zeros(R,nT,0) if no ecological variable
% - G: connectivity matrix for cultural diffusion (square matrix of size R)
% - nA: number of artifacts for each region and time point (matrix of
% integers of size R x nT)
%  - x0: probability of up-value for initial value of cultural trait (regions
% created de novo)
% - b: vector of transition [b_up b_down]
% - lambda: susceptibility to ecological factor
% - xi: cultural diffusion parameter
% - C: cultural artifacts parameter (vector of two element: probability of
% artifact of value 1 for T=down, probability of artifact of value 1 for
% T:up)
% - origin: region of origin for each region (vector of length R, use 0 for
% regions created de novo)
%
% OUTPUT ARGUMENTS
% - A: cultural artifacts: R x nT x 2 matrix with numbers of artifacts for
% artifacts of value 0 in A(:,:,1) and artifacts of value 1 in A(:,:,2)
% - T: cultural traits (R x nT matrix)
%
% See also etc_fit_binary, etc_rnd

[R, nT] = size(E);  %number of regions and number of time steps

onperiod = ~any(isnan(E),3);
starttime = zeros(1,R);
stoptime = zeros(1,R);
for r=1:R
    starttime(r) = find( onperiod(r,:),1); % initial point for each region
    stoptime(r) = find( onperiod(r,:),1,'last'); % initial point for each region
end
if nargin<10 % by default all regions start from scratch (x0 probability of being in up-state)
    origin = zeros(1,R);
end


T = zeros(R,nT); % cultural trait
A = zeros(R,nT); % cultural artifacts

% initial value of cultural triat
tt = nan(R,1);
tt(onperiod(:,1)) = (rand(sum(onperiod(:,1)),1)<x0); % set initially active regions to up-state with proba x0


for i=1:nT % for each time step
    active = onperiod(:,i);
    
    
    if i>1
        % newly active regions
        for r= find(active & ~onperiod(:,i-1))'
            if origin(r)>0 % herited from other region
                tt(r) = tt(origin(r));
            else % de novo
                tt(r) = rand<x0;
            end
        end
        
        % newly inactive regions
        tt(~active & onperiod(:,i-1)) = nan;
    end
    
    % output observable first
    for r= find(active)' %1:nreg % for each active region
        A(r,i) = sum(rand(1,nA(r,i))<C(1+tt(r))); %output var
    end
    
    RR = zeros(1,R);
    for r=find(active)' % for each region
        % then compute transition of states
        
        %%% !!!  should change that formula
        R(:,r) = B' + lambda'*E(r,i) + xi*G(r,active)*(2*tt(active)-1)*[1;-1]; % rate of transition (from 0 to 1 and 1 to 0, resp)
        RR(r) = R(tt(r)+1,r); % for current state
    end
    
    RR(RR<0) = 0; % no negative rates
    
    still = 1;
    left = dt(i); % how much time left
    while still % keep iterating as long as one trait changes value
        pt = - log(rand(1,R)) ./ RR; % generate Poisson time for transition for each region
        if any(pt<left) % if any transition occurs before end of time bin
            [ppt, r] = min(pt); % first state to transition
            tt(r) = 1-tt(r); % state transition
            left = left - ppt;
            R(:,active) = R(:,active) +  2*xi*(2*tt(r)-1)*[1;-1]*G(r,active);
        else % no more cultural trait is changed in time bin
            still = 0;
        end
    end
    
    T(:,i) = tt;
end

A = cat(3,nA-A, A); % number of observations of 0 and 1

