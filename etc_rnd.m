function [A,T] = etc_rnd(dt,E, G, nA, rho, lambda, xi, sigma, gamma, x0, sigma0, origin, obs, epsilon )
% generates cultural traits and artifacts from Evoked and Transmitted
% Culture (ETC) model with continuous trait.
%
% [A,T] = etc_rnd(dt,E, G, nA, rho, lambda, xi, sigma, gamma )
% - dt: time step (scalar)
% - E: ecological variable in each region and each time step (R x nT matrix,
% where R is the number of regions and nT the number of time steps). Use
% zeros(R,nT,0) if no ecological variable
% - G: connectivity matrix for cultural diffusion (square matrix of size R)
% - nA: number of artifacts for each region and time point (matrix of
% integers of size R x nT)
% - gamma: bias in evolution of cultural trait
% - lambda: susceptibility to ecological factor
% - xi: cultural diffusion parameter
% - rho: cultural leak
% - sigma: variance of noise in evolution of cultural trait
%
% A: cultural artifacts (see format below)
% T: cultural traits (R x nT matrix)
%
% [A,T] = etc_rnd(dt,E, G, nA, rho, lambda, xi, sigma, gamma, x0 )
% - x0: mean value of prior for initial value of cultural trait (regions
% created de novo)
%
% [A,T] = etc_rnd(dt,E, G, nA, rho, lambda, xi, sigma, gamma, x0, sigma0 )
% - sigma0: variance of prior for initial value of cultural trait
%
%  [A,T] = etc_rnd(dt,E, G, nA, rho, lambda, xi, sigma, gamma, x0, sigma0)
% - origin: region of origin for each region (vector of length R, use 0 for
% regions created de novo)
%
% [A,T] = etc_rnd(dt,E, G, nA, rho, lambda, xi, sigma, gamma, x0, sigma0, origin, obs )
% type of observations (possible value: 'binary'[default],'direct','continuous','count')
% Depending on obs, format of A will be: 
%      - direct observation: R x nT matrix (use nan for unobserved point)
%      - binary observation: R x nT x 2 matrix with numbers of artifacts for
%      artifacts of value 0 in A(:,:,1) and artifacts of value 1 in
%      A(:,:,2)
%      - count/continuous observation: R x nT cell array with value of
%      artifacts for each region and time step in each cell
%
% [A,T] = etc_rnd(dt,E, G, nA, rho, lambda, xi, sigma, gamma, x0, sigma0, origin, 'continuous', epsilon )
% - epsilon: variance of observation noise
%

[R, nT] = size(E); %number of regions and time steps

% initialize parameters
if nargin<10 || isempty(x0)
    x0 = 0; % mean initial value of latent variable
end
if nargin<11 || isempty(sigma0)
    sigma0 = 1;  % variance of initial value of latent variable
end
if nargin<12 || isempty(origin)
    origin = zeros(1,R);
end
if nargin<13
    obs = 'binary';
end

n_denovo = sum(origin==0); % number of regions de novo
if length(x0) ==1, x0 = x0*ones(1,n_denovo); end % one value per de novo region
x0all = zeros(R,1);
x0all(origin==0) = x0;

% dynamics of cultural trait
T = zeros(R,nT+1); % latent variable (cultural trait)
switch lower(obs)
    case 'binary'
        A = zeros(R,nT,2); % 3D matrix of observables (first matrix: non-romantic book, second: romantic books)
    case {'count','continuous'}
        A = cell(R,nT); % cell array with count for each observable
    case 'direct'
        A = nan(R,nT);
    otherwise
        error('incorrect observation type');
end
in = ~isnan(E(:,1)); % active regions for that time step
T(in,1) = x0all(in) + sqrt(sigma0)*randn(sum(in),1); % initial value

for t=1:nT
    in = ~isnan(E(:,t)); % active regions for that time step
    n_in = sum(in);
    
    Gin = G(in,in);
    Gtilde = Gin - diag(sum(Gin,2)); %connectivity matrix with negative terms along diagonal
    M = (1-rho*dt)*eye(n_in)+dt*xi*Gtilde; % auto-regressive matrix
    
    T(in,t+1) =  M*T(in,t) + dt*lambda*E(in,t) + dt*gamma + sqrt(sigma*dt)*randn(n_in,1); % first-order auto-regressive model (eq 2.2)
    
    if t<nT
        starting = find(~in &   ~isnan(E(:,t+1))); % regions starting at this time point
        for r=starting'
            if origin(r)>0
                T(r,t+1) = T(origin(r),t+1); % herited from another region
            else
                T(r,t+1) = x0all(r) + sqrt(sigma0)*randn; % initial value
            end
        end
    end
    
    %  if t>1
    %     stopping =
    %  end
    
    % yplus = binornd(nobs(:,t),pp); % draw number of books for each region from Bernoulli distribution
    switch lower(obs)
        case 'binary'
            pp = zeros(R,1);
            pp(in) = 1./(1+exp(-T(in,t+1))); % probability that each book is romantic (logistic)
            
            yplus = zeros(R,1);
            for r= find(in)'
                yplus(r) = sum(rand(nA(r,t),1)<pp(r));
            end
            A(in,t,1) = nA(in,t)-yplus(in);
            A(in,t,2) = yplus(in);
        case 'count'
            for r=find(in)'
                A{r,t} = poissrand( exp(T(r,t+1)), nA(r,t)); % generate counts from poisson distribution
            end
        case 'continuous'
            for r=find(in)'
                A{r,t} = normrnd( T(r,t+1), epsilon, 1, nA(r,t)); % generate counts from poisson distribution
            end
        case 'direct'
            for r=find(in)'
                if nA(r,t)
                    A(r,t) = T(r,t+1);
                end
            end
    end
    
    
end
end


function K = poissrand(lambda, n)
%  K = poissrand(lambda, n)
% generate n sample from Poisson distribution with rate lambda
%  (Knuth algorithm)
%
% See also poissrnd

L = exp(-lambda);

K = zeros(1,n);

for i=1:n
    k = 0;
    p = 1;
    while p>L
        k = k + 1;
        p = p*rand;
    end
    
    K(i) = k-1;
end
end