function [gamma,lambda, xi, rho, sigma, x0, sigma0, output] = etc_fit(dt, E, G, A, par_ini, options, origin )
% [gamma,lambda, xi, rho, sigma, x0, sigma0, output] = etc_fit(dt, E, G, Y, par_ini, options, origin )
% fits Evoked and Transmitted Culture (ETC) models with continuous cultural traits.
%
% INPUT ARGUMENTS:
% - dt: time step (scalar)
% - E: ecological variable in each region and each time step (R x T matrix,
% where R is the number of regions and T the number of time steps). Use []
% if no ecological variable
% - G: connectivity matrix for cultural diffusion (square matrix of size R)
% - A: artifacts data. Depending on type of artifactual data, format should be:
%      - direct observation: R x T matrix (use nan for unobserved point)
%      - binary observation: R x T x 2 matrix with numbers of artifacts for
%      artifacts of value 0 in A(:,:,1) and artifacts of value 1 in
%      A(:,:,2)
%      - count/continuous observation: R x T cell array with value of
%      artifacts for each region and time step in each cell
%
% - par_ini: initial value of parameters (vector for [gamma,lambda, xi,
% rho, sigma, x0, sigma0]). Use [] for default initial value
%
% - options: structure with options for fitting. Possible fields are:
%        - 'observations': type of observations (possible value: 'binary','direct','continuous','count')
%        - 'maxiter': maximum number of iterations for EM algorithm (default: 200)
%        - 'TolFun': tolerance value for stopping EM algorithm (default
%        0.005)
%        - 'initialpoints': number of initial sets of parameter values for EM algorithm (default: 50)
%        - 'bootstrap': number of parametric boostraps used to compute standard errors over parameters (default:0).
%        - 'GP': method for approximate E step: 'moment'(default),'EP' or 'Laplace'
%        - 'diffusion': boolean representing whether model includes cultural diffusion (default: true)
%        - 'fixedinitialspread': boolean representing whether sigma0^2 is fixed or fitted (default:true)
%        - 'individualinitialpoint': boolean representing whether each region created de novo uses prior with same mean (default: false)
%        - 'verbose': sets verbosity of algorithm: 'on'(default),'off','little'
%        - 'trainset' and 'testset': training and test set(s) for
%        crossvalidation. Use cell array of artifacts indices (over the
%      total number of artifacts).
% - origin: region of origin for each region (vector of length R, use 0 for
% regions created de novo)
%
% OUTPUT PARAMETERS:
%
% gamma: bias in evolution of cultural trait
% lambda: susceptibility to ecological factor
% xi: cultural diffusion parameter
% rho: cultural leak
% sigma: variance of noise in evolution of cultural trait
% x0: mean value of prior for initial value of cultural trait (regions
% created de novo)
% sigma0: variance of prior for initial value of cultural trait
% output: structure with a lot of fields (most should be self-explanatory)
%
% See also etc_rnd, etc_test, etc_fit_binary

% default parameters
observations = 'binary'; % type of observations ('continuous', 'binary','count' or 'direct')
initialpoints = 50; % number of initial points in the minimization procedure
maxiter = 200;
maxiter_bts = 50; % maximum number of iterations for bootstrapping
maxiter_xv = 50; % maximum number of iterations for cross-validation
TolFun = .005;
GPalgo = 'moment'; % which algorithm to use for GP approximation (E step): moment (Smith'03), 'Laplace' or 'EP'
diffusion = 1; % whether we include geographical diffusion
crossvalid = 0; % evaluate cross-validated LLH
fixedinitialspread = 1; % 0 to include parameter sigma0 for initial spread, 1 if fixed to 1
individualinitialpoint = 0; % 1 to have one initial point parameter per region, 0 to have one single parameter
bootstrap = 0;

if nargin < 6
    options = struct;
end

[R, T,~] = size(E);  %number of regions x number of time steps x n external factors

if nargin<7 % default: all regions created from scratch
    origin = zeros(1,R);
end

with_ext = ~isempty(E); % whether there is any external variable
if ~with_ext
    E = zeros(R,T);
end

% period for each region
if isfield(options, 'onperiod')
    onperiod = options.onperiod;
else
    onperiod = ~any(isnan(E),3);
end


if ~isfield(options,'observations')
    options.observations = observations;
else assert(any(strcmpi(options.observations, {'continuous','binary','count','direct'})));
end
if ~isfield(options, 'maxiter') % maximum iterations of the EM algo
    options.maxiter = maxiter;
end
if ~isfield(options, 'TolFun')
    options.TolFun = TolFun;
end
if ~isfield(options, 'initialpoints')
    options.initialpoints = initialpoints;
end
if isfield(options, 'bootstrap')
    bootstrap = options.bootstrap;
end
if ~isfield(options, 'GP')
    options.GP = GPalgo;
end

continuous_obs = strcmpi(options.observations, 'continuous');
if continuous_obs && ~strcmpi(options.GP, 'moment')
    fprintf('gaussian observations, using moment algo which provides exact inference faster')
    options.GP = 'moment';
end
if ~isfield(options, 'diffusion')
    options.diffusion = diffusion;
end
if ~isfield(options, 'fixedinitialspread')
    options.fixedinitialspread = fixedinitialspread;
end
if ~isfield(options,'individualinitialpoint')
    options.individualinitialpoint = individualinitialpoint;
end
if ~isfield(options, 'verbose')
    options.verbose = 'on';
end
if isfield(options, 'trainset')
    crossvalid = 1;
    assert(isfield(options, 'testset'), 'for crossvalidation, you should provide both fields trainset and testset');
end
options.with_ext = with_ext;


switch lower(options.GP)
    case 'moment'
        initialpoints_bts = min(initialpoints,10);
        initialpoints_xv = min(initialpoints,10);
    case 'ep'
        initialpoints_bts = 2;
        initialpoints_xv = 2;
    case 'laplace'
        initialpoints_bts = 2; 20;
        initialpoints_xv = 2;
    otherwise
        error('incorrect GP inference: %s', options.GP);
end


if any(~any(onperiod))
    error('there should not be time point with no active dataset');
end
E(~onperiod) =0;
%ndatapoint = sum(onperiod(:)); % total number of time points

% number of observations (artifacts)
Aon = reshape(A,R*T,1 + strcmpi(options.observations,'binary')); % merge all observations across regions
Aon = Aon(onperiod,:); % observations points only for active regions
switch options.observations
    case 'binary'
        nA = sum(Aon(:)); % total number of observations
        nA_datapoint = sum(A,3); % number of observation per data point
    case {'count','continuous'}
        nA = sum(cellfun(@length, Aon));
        nA_datapoint = cellfun(@length, A);
    case 'direct'
        nA = sum(~isnan(Aon(:)));
        nA_datapoint = ~isnan(A)*1;
end

onperiod(:,end+1) = onperiod(:,end); % we add final row for last time step


if options.individualinitialpoint
    n_x0 = sum(origin==0); % one parameter per region which is created de novo
else
    n_x0 = 1; % one single parameter
end
options.n_x0 = n_x0;

% initialize parameters
if nargin<=3 || isempty(par_ini)  % default initial parameters
    rho = 5/T; %leak
    lambda = 0; %dependence on stimulus
    xi = 0; % dependence on neighbouring regions
    sigma = 1*.01/dt; % variance of auto-regressive noise (! here sigma stands for sigma^2 in equations)
    gamma = 0; % overall offset
    x0 = zeros(1,n_x0); % initial value of latent variable
    sigma0 = 1;
    par_ini = [gamma lambda xi rho sigma x0 sigma0];
    if ~options.diffusion
        par_ini(3) = [];
    end
    if ~with_ext
        par_ini(2) = [];
    end
    if options.fixedinitialspread
        par_ini(end) = [];
    end
    if continuous_obs
        epsilon = mean([A{:}].^2); % variance of observation noise for gaussian observation
        par_ini(end+1) = epsilon;
    end
end

npar = 4 + with_ext + options.diffusion +n_x0 -options.fixedinitialspread + continuous_obs; % total number of free parameters
options.npar = npar;

freepars = [true with_ext logical(options.diffusion) true(1,2+n_x0) ~logical(options.fixedinitialspread) true(1,continuous_obs)]; % which parameters are free (depending if external variable and diffusion are enabled)


%% EM algorithm with multiple initial points
[par_hat, output] = EMalgo_multinit(dt, R,T,  par_ini, E,  A, G, onperiod, origin, options);

[gamma,lambda,xi,rho,sigma,x0,sigma0,epsilon] = par_decomp(par_hat,with_ext,options);

output.par_hat = [gamma,lambda,xi,rho,sigma,x0,sigma0,epsilon]; % estimated parameters
output.label =  {'\gamma' '\lambda','\xi','\rho','\sigma','x_0','\sigma_0','\epsilon'}; % labels for each parameters

output.nObs = nA; % number of observations
output.nDatapoint = sum(onperiod(:)); % number of datapoints

if bootstrap>0
    
    %% parametric bootstrapping (Visser et al 2000)
    par_bts = zeros(bootstrap, npar);
    
    options_bs = options;
    options_bs.bootstrap = 0;
    options_bs.maxiter = maxiter_bts;
    options_bs.initialpoints = initialpoints_bts;
    options_bs.verbose = 'off';
    options_bs.maxiter = maxiter_xv;
    
    
    Ebts_gen = E;
    Ebts_gen(~onperiod(:,end-1)) = nan;
    obs = options.observations;
    
    fprintf('RUNNING BOOTSTRAPPING');
    
    parfor b=1:bootstrap % for each permutation
        %        parfor b=1:bootstrap % for each permutation
        
        fprintf('.');
        
        % generate sample from best-fitting values
        Abs = etc_rnd(dt,Ebts_gen, G, nA_datapoint, rho, lambda, xi, sigma, gamma, x0, sigma0, origin, obs, epsilon );
        
        % compute best-fitting parameters for this values
        [par_bts(b,:),output_bts(b)]= EMalgo_multinit(dt, R,T,  par_hat, E,  Abs, G, onperiod, origin, options_bs);
        fprintf('*');
    end
    fprintf('done\n');
    output_bts = rmfield(output_bts,{'x_tT','Wt','W_diff','W_tT'}); % those fields take a lot of space
    output.par_bts = zeros(bootstrap,7);
    if options.fixedinitialspread
        output.par_bts(:,end) = 1;
    end
    output.par_bts(:,freepars) = par_bts;
    output.mean_bts = mean(output.par_bts,1);
    oneside = min( mean(output.par_bts>=0), mean(output.par_bts<=0)); % smaller tail for bottstrap samples
    output.p_bts = min(2*oneside,1); % p-values from bts
    output.se_bts = std(output.par_bts,1);
    output.output_bts = output_bts;
else
    output.par_bts = [];
    output.mean_bts = [];
    output.p_bts = [];
    output.se_bts = [];
    output.output_bts = [];
end


%% cross-validation
if crossvalid
    if ~iscell(options.trainset)
        options.trainset = {options.trainset};
        options.testset = {options.testset};
    end
    
    n_xvalid = length(options.trainset); % number of sets
    CVLL = nan(1,n_xvalid); % crossvalidated log-likelihood
    options_xv = options;
    options_xv.verbose = 'off';
    options_xv.initialpoints = initialpoints_xv;
    
    fprintf('RUNNING CROSSVALIDATION');
    parfor v=1:n_xvalid
        
        % transform vectors of training and test sets into number of
        % observable for each region and time
        Atrain = observation_subset(A, options.trainset{v});
        Atest = observation_subset(A, options.testset{v});
        
        % estimate parameters with training set set
        par_xv = EMalgo_multinit(dt, R,T,  par_hat, E,  Atrain, G, onperiod, origin, options_xv);
        
        [gamma_xv,lambda_xv,xi_xv,rho_xv,sigma_xv,x0_xv,sigma0_xv, epsilon_xv] = par_decomp(par_xv,with_ext,options);
        
        % now Estep and Mstep to compute change in parameters
        Gxi_xv = xi_xv*G;
        I_xv = dt*(lambda_xv*E+gamma_xv); % overall input onto each region
        
        % compute LLH for test set using fitted parameters
        [~, ~, ~, ~, CVLL(v)] = Estep(dt, R,T,  sigma_xv, x0_xv, sigma0_xv, I_xv, rho_xv,Gxi_xv, Atest,onperiod, origin,options, epsilon_xv);
        
        
        fprintf('o');
    end
    fprintf('done\n');
    output.CVLL = CVLL;
    output.meanCVLL = mean(CVLL);
else
    output.CVLL = [];
    output.meanCVLL = [];
end


%% compute influence of vertical transmission, horiztonal transmission and evoked component
Tgeneration = .25; % 25 years generation
infl = influence_analysis(T, Tgeneration, gamma,lambda,xi,rho,sigma, E, G, onperiod, output.x_tT);
output.influence_vertical_transmission = infl(1);
output.influence_horizontal_transmission = infl(2);
output.influence_evoked = infl(3);

%% Laplace approximation
fprintf('computing hessian ...');
[~, ~,  hess_Mstep] = Mstep(dt,R,T, output.x_tT, output.Wt, output.W_diff, E, G, onperiod, A, output.W_tT, options, origin);

%then compute dM/dpar using finite difference
dx = 1e-3;
MM = zeros(npar);
for p=0:npar % first one is one more E and M step, further ones are E+M with infinitesimal change along one dim
    par_dx = par_hat + dx*((1:npar)==p); % change one parameter by dx
    [gamma,lambda,xi,rho,sigma,x0,sigma0,epsilon] = par_decomp(par_dx,with_ext,options);
    
    % now Estep and Mstep to compute change in parameters
    Gxi = xi*G;
    I = dt*(lambda*E+gamma); % overall input onto each region
    [x_tT_dx, Wt_dx, W_diff_dx, W_tT_dx] = Estep(dt, R,T,sigma,x0, sigma0, I,  rho, Gxi, A,onperiod, origin,options, epsilon);
    par_dx2 = Mstep(dt,R,T, x_tT_dx, Wt_dx, W_diff_dx, E, G, onperiod, A, W_tT_dx, options, origin);
    if p==0
        par_hat_plusone = par_dx2;
    else
        MM(p,:) = (par_dx2 - par_hat_plusone)/dx;
    end
end

hess = hess_Mstep * (eye(npar)-MM); % Hessian in EM, see. e.g. Jamshidian & Jennrich 2000, equation 1
output.hess = hess;

output.covb = inv(-hess); %Covariance matrix

% standard error of estimates
output.se = sqrt(diag(output.covb))';

% T-statistic for the weights
output.T = par_hat ./ output.se;

% p-value for significance of each coefficient (Wald test), two-tailed
output.p = 2*normcdf(-abs(output.T));

% standard error, T and p for all parameters (including fixed)
isfreepar = [true with_ext options.diffusion true(1,2+options.n_x0) ~options.fixedinitialspread continuous_obs];
npartot = 6+options.n_x0+continuous_obs;
output.se_all = zeros(1,npartot);
output.se_all(isfreepar) = output.se;
output.T_all = nan(1,npartot);
output.T_all(isfreepar) = output.T;
output.p_all = nan(1,npartot);
output.p_all(isfreepar) = output.p;

% model evidence
output.logevidence = output.LLH + npar/2*log(2*pi) - log(det(-hess))/2; % Laplace approximation
output.BIC = npar*log(nA) -2*output.LLH; % Bayes Information Criterion
output.AIC = 2*npar - 2*output.LLH; % Akaike Information Criterior
output.AICc = output.AIC + 2*npar*(npar+1)/(nA-npar-1); % AIC corrected for sample size


% go back to best-fitting parameters
[gamma,lambda,xi,rho,sigma,x0,sigma0,output.epsilon] = par_decomp(par_hat,with_ext,options);


end
%% end of main function

%% EM algo multiple starting points
function [par_hat, output] = EMalgo_multinit(dt, R,T,  par_ini, E,  A, G, onperiod, origin, options)

with_ext = options.with_ext;
with_diffusion = options.diffusion;
fixedinitialspread = options.fixedinitialspread;
initialpoints = options.initialpoints;

npar = options.npar; % total number of free parameters
par_hat_all = zeros(npar, initialpoints);
LLHhist = cell(1,initialpoints);
par_hat_hist = cell(1,initialpoints);
LLH_all = zeros(1,initialpoints);

spmd
    warning('off','MATLAB:nearlySingularMatrix');
    warning('off','MATLAB:lang:cannotClearExecutingFunction');
end
warning('off','MATLAB:illConditionedMatrix');

for init=1:initialpoints % for each value of initial points
    
    
    %% initialize parameters
    if init==1
        % first initial point : default value
        [gamma,lambda,xi,rho,sigma,x0,sigma0,epsilon] = par_decomp(par_ini,with_ext,options);
    else
        %% random initial parameters
        rho = 1*rand/T/dt; %auto-regression model (between -0.05 and 0.05)
        if with_ext
            lambda = .1*randn/mean(abs(E(:)))/T/dt; %dependence on stimulus
        else
            lambda = 0;
        end
        Gtilde = G - diag(sum(G,2)); %connectivity matrix with negative terms along diagonal
        xi = with_diffusion * .2*(rand-.5)/dt/T/eigs(Gtilde,1); % dependence on neighbouring regions
        sigma = .5*rand/T/dt; % variance of auto-regressive noise (! here sigma stands for sigma^2 in equations)
        gamma = randn/dt/T; % bias in state update equation
        x0 = randn(1,options.n_x0); % initial value of latent variable
        if fixedinitialspread
            sigma0 =1;
        else
            sigma0 = rand; % variance for initial value of latent
        end
        if strcmpi(options.observations, 'continuous')
            epsilon = 2*rand*mean([A{:}]);
        end
    end
    
    %% run EM algo
    [par_hat_all(:,init), ~, ~, ~, ~, LLH_all(init), LLHhist{init}, par_hat_hist{init}] = EMalgo(dt, R,T,  rho, sigma, gamma, lambda,xi, x0, sigma0,...
        E,  A, G, onperiod, origin,options, epsilon);
    
    if any(strcmp(options.verbose, {'on','little'}))
        fprintf('Starting point %d/%d, %d EM iterations, LLH: %f\n',init, initialpoints, length(LLHhist{init}), LLH_all(init));
    end
    
end

% select best value
[LLH, init] = max(LLH_all);
par_hat = par_hat_all(:,init)';

output = struct;
output.LLH_all = LLH_all;
output.LLHhist = LLHhist{init};
output.LLHhist_all = LLHhist;
output.par_hat_all = par_hat_all;
output.par_hat_hist = par_hat_hist;

% use E-step to compute posterior distribution over latent
[gamma,lambda,xi,rho,sigma,x0,sigma0,epsilon] = par_decomp(par_hat,with_ext,options);
Gxi = G*xi;
I = dt*(lambda*E+gamma); % overall input onto each region
[output.x_tT, output.Wt, output.W_diff, output.W_tT, output.LLH] = Estep(dt, R,T,  sigma, x0, sigma0, I, rho, Gxi, A,onperiod, origin,options, epsilon); % compute posterior over latent

for r=1:R
    output.x_se(r,:) = sqrt(output.W_tT(r,r,:));
end

% posterior mean and covariance: assign nan when region is not
% active
onperiod2 = onperiod | [false(R,1) onperiod(:,1:end-1)]; % active at t or t+1
output.x_tT(~onperiod2) = nan;
output.x_se(~onperiod2) = nan;

end

%% EM algorithm
function [par_hat, x_tT, Wt, W_diff, W_tT, LLH, LLHhist,x_hist] = EMalgo(dt, R,T,  rho, sigma, gamma, lambda,xi, x0, sigma0, E,  A, G, onperiod, origin,options, epsilon)

maxiter = options.maxiter;
iter = 0; %EM iteration

Gxi = G*xi;
I = dt*(lambda*E+gamma); % overall input onto each region

%initialize stuff
LLH = -Inf;
oldLLH = nan;
LLHhist = zeros(1,maxiter);
nanllh = false;  % stop if reaches nan LLH

continuous_obs = strcmpi(options.observations, 'continuous');

%history of parameter values across iterations of EM
x_hist(:,1) =  [gamma,lambda,xi,rho,sigma,x0,sigma0,epsilon];
if ~options.diffusion, x_hist(3) = []; end
if ~options.with_ext, x_hist(2) = []; end
if options.fixedinitialspread, x_hist(end-continuous_obs) = []; end
if ~continuous_obs, x_hist(end) = []; end

%% EM loop
while (iter<2 || ((LLH-oldLLH)>options.TolFun && iter<maxiter)) && ~nanllh % loop until gain in LLH is small enough, reached max number of iterations or nan value for LLH
    
    oldLLH = LLH;
    
    % E-step
    [x_tT, Wt, W_diff, W_tT, LLH] = Estep(dt, R,T,sigma, x0, sigma0, I, rho, Gxi, A,onperiod, origin,options,epsilon);
    
    %% M-step
    if ~isnan(LLH) && ~isinf(LLH)
        [par_hat, I] = Mstep(dt,R,T, x_tT, Wt, W_diff, E, G, onperiod, A, W_tT, options, origin);
        [gamma,lambda,xi,rho,sigma,x0,sigma0,epsilon] = par_decomp(par_hat,options.with_ext,options);
        Gxi = G*xi;
        
        iter = iter+1;
        LLHhist(iter) = LLH;
        x_hist(:,iter+1) = par_hat;
    else % if reached nan or inf value
        nanllh = true;
        if iter ==0
            par_hat = nan(1,options.npar);
        end
        
    end
    
    if strcmp(options.verbose, 'on')
        fprintf('iter %d, diff LLH %f\n',iter, LLH-oldLLH);
    end
end
LLHhist(iter+1:end) = [];

x_hist(:,iter+1:end) = [];
end


%% E-step
function [x_tT, Wt, W_diff, W_tT, LLH] = Estep(dt, R,T,  sigma, x0, sigma0, I, rho, Gxi, A,onperiod, origin,options, epsilon)

binary_obs = strcmpi(options.observations, 'binary');
direct_obs = strcmpi(options.observations, 'direct');
if options.n_x0 ==1
    n_denovo = sum(origin==0);
    x0 = x0*ones(1,n_denovo); % one value for all regions created de novo
end
x0all = zeros(R,1);
x0all(origin==0) = x0;

if ~strcmpi(options.GP,'moment') || direct_obs
    %% Laplace or EP method
    
    
    
    onperiod2 = onperiod | [false(R,1) onperiod(:,1:end-1)]; % active at t or t+1
    for r=1:R
        if origin(r)>0
            t = 1+ find(~onperiod(r,1:end-1) & onperiod(r,2:end)); % starting point
            onperiod2(r,t) = 0; % remove data point from GP to keep covariance matrix definite positive
        end
    end
    nDT = sum(onperiod2,1); % number of active regions per time step
    n_datapoints2 = sum(nDT); %onperiod2(:));
    
    % X and A values for training set
    A2 = reshape(A, R*T,1+binary_obs);
    A2 = A2(onperiod(:,1:end-1),:);
    
    idx_post = [false(R,1) onperiod]; % active at t+1
    idx_post= idx_post(onperiod2);
    idxpost = find(idx_post); % subset of points active at t or t+1 that are active at t+1
    if binary_obs % binary observations
        xdata = [repelem(idxpost, A2(:,1)) ; ...
            repelem(idxpost, A2(:,2))]; % 2D coordinates for observed points
        Adata = [-ones(sum(A2(:,1)),1); +ones(sum(A2(:,2)),1)]; % observable (-1/+1)
        nA = sum(A2(:)); % total number of observations
    elseif direct_obs % direct observations
        xdata = idxpost(~isnan(A2));
        Adata = A2(~isnan(A2));
        nA = length(xdata);
    else % counts/continuous observations
        xdata = repelem(idxpost, cellfun(@length, A2));
        Adata = [A2{:}]';
        nA = sum(cellfun(@length,A2(:))); % total number of observations
    end
    
    % compute prior mean
    mu = zeros(R,T+1);
    
    % initial point
    active = onperiod(:,1);
    mu(active,1) = x0all(active); % initial value x_{0|0}
    
    % compute prior mean iteratively
    for t=1:T
        active = onperiod(:,t); % regions active for that time step
        in = find(active); % regions active t+1
        if t>1
            starting = ~onperiod(:,t-1) & active;
            
            % initial point for a region:
            for r= find(starting)'
                if origin(r) % if coming from scission of region
                    assert( any(in==origin(r)));
                    mu(r,t) = mu(origin(r),t);
                else %starting from scratch
                    mu(r,t) = x0all(r);
                end
            end
        end
        
        M_in = ARmatrix(Gxi, in, rho, dt); % auo-regressive matrix
        mu(in,t+1) = M_in*mu(in,t) +I(in,t);
    end
    
    
    if direct_obs %to minimize memory we only store the covariance we need
        covx = zeros(nA,nA); % prior covariance for observation points
        cov_cross = zeros(nA, n_datapoints2); % cross-covariance between observation points and all active points
        cov_auto = cell(1,R+1); % covariance for each time step (and with following time step)
    else
        cov = zeros(n_datapoints2,n_datapoints2);
    end
    
    % compute prior covariance iteratively
    % see formula for covariance of MAT(1) process in https://stats.stackexchange.com/questions/171304/covariance-in-ar1-process
    active = onperiod(:,1); % regions active for that time step and subsequent one
    SS = sigma0*diag(active);
    cnt_t = 0;
    for t=1:T+1
        
        % for new regions
        if t>1
            starting = find(~onperiod(:,t-1) & onperiod(:,t));
            for r=starting'
                if origin(r)>0
                    assert( any(in==origin(r)), 'region of origin not active');
                    SS(r,:) = SS(origin(r),:);
                    SS(:,r) = SS(:,origin(r));
                    % onperiod2(r,t) = 0; % remove data point from GP to keep covariance matrix definite positive
                    
                else
                    SS(r,r) = sigma0;
                end
            end
        end
        
        % compute cross_covariance terms
        Mv = eye(R);
        idx_t = cnt_t + (1:nDT(t));
        cnt_u = cnt_t;
        if direct_obs
            n_id = sum(nDT(t:min(t+1,T+1)));  % number of datapoints to process (two time steps except for last)
            cov_auto{t} = zeros(n_id);
        end
        
        for u=t:T+1
            active = onperiod(:,u); % regions active for that time step and subsequent one
            in = find(active); % regions active t+1
            
            % starting region
            if (u>1) && (u>t)
                starting = find(~onperiod(:,u-1) & active & origin');
                for r=starting'
                    org = origin(r);
                    assert(any(in==org),'region of origin not active');
                    Mv(r,:) = Mv(org,:);
                end
            end
            
            this_cov = Mv * SS; % cross-covariance between time steps
            this_cov = this_cov(onperiod2(:,u),onperiod2(:,t));  % select only active regions for each time step
            %    cov(:,:,u,t) = Mv * SS;
            %    cov(:,:,t,u) =  cov(:,:,u,t)';
            
            idx_u = cnt_u + (1:nDT(u));
            if direct_obs
                if t==u
                    iii = 1:nDT(t);
                    cov_auto{t}(iii, iii) = this_cov;
                    if t>1
                        iii = nDT(t-1)+(1:nDT(t));
                        cov_auto{t-1}(iii, iii) = this_cov;
                    end
                elseif u==t+1
                    iii = 1:nDT(t);
                    jjj = nDT(t)+(1:nDT(u));
                    cov_auto{t}(jjj,iii) = this_cov;
                    cov_auto{t}(iii,jjj) = this_cov';
                    
                end
                % covariance for observation points
                isobs_t= ismember(idx_t, xdata);
                isobs_u = ismember(idx_u, xdata);
                if  any(isobs_u)
                    ii_u = ismember(xdata, idx_u(isobs_u));
                    cov_cross(ii_u, idx_t) = this_cov(isobs_u,:);
                end
                if any(isobs_t)
                    ii_t = ismember(xdata, idx_t(isobs_t));
                    cov_cross(ii_t, idx_u) = this_cov(:,isobs_t)';
                    if  any(isobs_u)
                        covx( ii_u,  ii_t) = this_cov(isobs_u,isobs_t);
                        covx( ii_t,  ii_u) = this_cov(isobs_t,isobs_u);
                        
                    end
                end
                
            else
                cov(idx_u,idx_t) = this_cov;
                cov(idx_t,idx_u) = this_cov';
            end
            
            M_in = zeros(R);
            M_in(in,in) = ARmatrix(Gxi, in, rho, dt); % auo-regressive matrix
            Mv = M_in*Mv; % M^(u-t)
            
            cnt_u = cnt_u + nDT(u);
            
        end
        
        % update covariance term
        active = onperiod(:,t); % regions active for that time step and subsequent one
        in = find(active); % regions active at t
        M_in = zeros(R);
        M_in(in,in) = ARmatrix(Gxi, in, rho, dt); % auo-regressive matrix
        
        SS = M_in*SS*M_in + dt*sigma*diag(active);
        
        % update counter for position in covariance matrix
        cnt_t = cnt_t +nDT(t);
        
    end
    
    % cov = reshape(permute(cov,[1 3 2 4]),R*(T+1),R*(T+1)); % convert to 2D matrix (datapoint x datapoint)
    
    % cov = cov(onperiod2,onperiod2); % select only active datapoints
    % n_datapoints2 = sum(onperiod2(:));
    
    % define structure for GPflow toolbox
    hyp = struct('mean', mu(onperiod2), 'lik', []);
    
    % test for definite positiveness and extract Cholevsky
    if ~direct_obs
        [L,nonSDP] = chol(cov);
        if nonSDP % try to add just a bit of independent noise to see if it helps
            fprintf('adding a bit of diagonal terms to have definite positive covariance\n');
            cov = cov + 1e-2*eye(n_datapoints2);
            [L,nonSDP] = chol(cov);
        end
        if nonSDP % if non definite positive, return nan (shouldn't happen)
            LLH = nan;
            x_tT = nan(R,T+1);
            W_tT = nan(R,R,T+1); % sigma^2_{t|T} (indices from 0 to K)
            Wt = nan(R,R,T+1);
            W_diff = nan(R,R,T);
            return;
        end
        
        
        L(1:(n_datapoints2+1):end) = log(diag(L));
        hyp.cov = L(triu(true(n_datapoints2))); % provide cholevsky decomposition of covariance
    end
    
    
    
    if direct_obs % observing cultural traits directly
        %% direct
        
        x_tT = zeros(R,T+1);
        W_tT = zeros(R,R,T+1); %(indices from 0 to T)
        Wt = zeros(R,R,T+1);
        W_diff = zeros(R,R,T);
        
        id1 = 0;
        
        mx = hyp.mean(xdata); % prior mean
        % covx = cov(xdata,xdata); % covariance for points with data
        Sdm = covx\(Adata-mx);
        Lx = chol(covx);
        
        % compute LLH
        logdettrm = -log(2*pi)*length(mx)-sum(log(diag(Lx)));         % Log-determinant term
        
        % Quadratic term
        Xctr = Adata-mx;  % centered X
        Qtrm = - Xctr'/covx * Xctr /2;
        LLH = Qtrm + logdettrm;
        
        for t=1:T+1 % for each time step
            
            n_id = sum(nDT(t:min(t+1,T+1)));  % number of datapoints to process (two time steps except for last)
            id = id1 + (1:n_id);               % corresponding indices
            
            % kss = cov(id,id);    % prior self-variance
            % Ks = cov(xdata,id);   % covariance between test set and data points
            
            kss = cov_auto{t};
            Ks = cov_cross(:,id);
            
            in = onperiod2(:,t); % regions active for previous or current time step
            n_active = nDT(t);
            
            % posterior mean for this time step
            x_tT(in,t) = mu(in,t) + Ks(:,1:n_active)'*Sdm;
            
            V = Ks' / Lx;
            this_fs = kss - V*V';
            this_fs = max(this_fs,0);   % remove numerical noise i.e. negative variances
            
            % posterior covariance for this time step
            W_tT(in,in,t) = this_fs(1:n_active,1:n_active);
            
            % posterior cov for this and next time step
            if t<T+1
                this_Wdiff = this_fs(1:n_active,n_active+1:n_id);
                in_next = onperiod2(:,t+1);
                W_diff(in,in_next,t) = this_Wdiff; % eq 2.21 (part 1)
            end
            
            id1 = id1 + n_active;
        end
        
        
        
    else % use GPflow for non-direct osbervations
        
        %  perform inference for observed points and infer for whole grid
        switch lower(options.GP)
            case 'laplace'
                inf_method =  'infLaplace';
            case 'ep'
                inf_method = 'infEP';
            otherwise
                error('unknown GP algorithm %s',  options.GP);
        end
        if binary_obs
            lklhd = 'likLogistic'; % use logistic likelihood for binary observations
        else
            lklhd = {'likPoisson','exp'}; % Poisson likelihood for count observations
        end
        try
            % run GP inference on GPML
            [nlZ, ~, post ] = gp(hyp, inf_method, {'meanDiscrete',n_datapoints2}, {'covDiscrete',n_datapoints2}, lklhd, xdata, Adata);
            if any(post.sW<0)
                warning('negative posterior variance!!');
            end
        catch % if the inference algorithm somehow fails
            LLH = nan;
            x_tT = nan(R,T+1);
            W_tT = nan(R,R,T+1); % sigma^2_{t|T} (indices from 0 to K)
            Wt = nan(R,R,T+1);
            W_diff = nan(R,R,T);
            return;
        end
        
        LLH = -nlZ;
        
        
        %% compute posterior mean, variance and covariance for consecutive data points)
        x_tT = zeros(R,T+1);
        W_tT = zeros(R,R,T+1); % sigma^2_{t|T} (indices from 0 to K)
        Wt = zeros(R,R,T+1);
        W_diff = zeros(R,R,T);
        [post.L,nonSDP2] = chol(diag(post.sW)*cov(xdata,xdata)*diag(post.sW) + eye(nA)); % cholevsky decomposition (Algo 3.2 in Rasmussen 2006)
        if nonSDP2
            fprintf('posterior is not definite positive:\n');
            fprintf('min eig of cov:%f\n', min(eig(cov)));
            fprintf('min eig of cov(xdata,xdata):%f\n', min(eig(cov(xdata,xdata))));
            fprintf('min eig of sW*cov(xdata,xdata)*sW:%f\n', min(eig(diag(post.sW)*cov(xdata,xdata)*diag(post.sW))) );
            
            LLH = nan;
            x_tT = nan(R,T+1);
            W_tT = nan(R,R,T+1); % sigma^2_{t|T} (indices from 0 to K)
            Wt = nan(R,R,T+1);
            W_diff = nan(R,R,T);
            return;
        end
        
        id1 = 0;
        for t=1:T+1 %
            
            n_id = sum(sum(onperiod2(:,t:min(t+1,T+1))));  % number of datapoints to process (two time steps except for last)
            id = id1 + (1:n_id);               % corresponding indices
            
            kss = cov(id,id); %diag(diag(cov(id,id)));              % prior self-variance
            
            Ks = cov(xdata,id);   % covariance between test set and data points
            
            in = onperiod2(:,t); % regions active for previous or current time step
            n_active = sum(in);
            
            x_tT(in,t) = mu(in,t) + Ks(:,1:n_active)'*post.alpha;        % conditional mean fs|f
            
            % L contains chol decomp => use Cholesky parameters (alpha,sW,L)
            if isnumeric(post.L)
                V  = post.L'\(repmat(post.sW,1,length(id)).*Ks); %  (Algo 3.2 in Rasmussen 2006)
                this_fs = kss - V'*V;                       % predictive variances
            else
                LKs = post.L(Ks); % matrix or callback
                this_fs = kss + diag(sum(Ks.*LKs,1)');
            end
            
            this_fs = max(this_fs,0);   % remove numerical noise i.e. negative variances
            
            W_tT(in,in,t) = this_fs(1:n_active,1:n_active);
            
            if t<T+1
                this_Wdiff = this_fs(1:n_active,n_active+1:n_id);
                in_next = onperiod2(:,t+1);
                W_diff(in,in_next,t) = this_Wdiff; % eq 2.21 (part 1)
            end
            
            id1 = id1 + n_active;
        end
        
    end
    
    % add values for region created from origin region at starting point
    for t=2:T
        starting = find( ~onperiod(:,t-1) & onperiod(:,t) & origin'>0); % regions starting at this time step and inheriting from other region
        for r=starting'
            onperiod2(r,t) = 1;
            org = origin(r);
            r_and_similar = intersect(starting,find(origin==org)); % region, and possibly other regions starting at same time from same region
            for s=r_and_similar'
                x_tT(s,t) = x_tT(org,t); % same posterior mean as region it inherits from
                W_diff(s,:,t) = W_diff(org,:,t);
                W_tT(s,:,t) = W_tT(org,:,t); %
            end
            for s=r_and_similar'
                W_tT(:,s,t) = W_tT(:,org,t); %
            end
            
        end
    end
    
    % compute second moments
    for t=1:T+1
        in = onperiod2(:,t);
        Wt(in,in,t) =  W_tT(in,in,t) + x_tT(in,t)*x_tT(in,t)'; % eq 2.22
        
        if t<T+1
            in_next = onperiod2(:,t+1);
            W_diff(in,in_next,t) =  W_diff(in,in_next,t) + x_tT(in,t)*x_tT(in_next,t+1)'; % eq 2.21 (part 2)
        end
    end
    
    
else
    %% 'moment' method
    
    % E-step I: non-linear recursive filter
    x_same = zeros(R, T+1); % value for x_{t|t} for k from 0 to T
    x_diff = zeros(R, T); % value for x_{t+1|t} for t from 0 to T-1
    Wsame = zeros(R, R, T+1); % value for W_{t|t} for t from 0 to K
    Wdiff = zeros(R, R, T); % value for W_{t+1|t} for t from 0 to T
    
    % initial point
    active = onperiod(:,1);
    x_same(active,1) = x0all(active); % initial value x_{0|0}
    Wsame(:,:,1) = sigma0*eye(R); %  % initial value sigma^2_{0|0}
    LLH = 0;
    
    for t=1:T
        active = onperiod(:,t); % regions active for that time step and subsequent one
        in = find(active); % regions active t+1
        %% newly active regions
        if t>1
            starting = ~onperiod(:,t-1) & active;
            
            % initial point for a region:
            for r= find(starting)'
                if origin(r) % if coming from scission of region
                    org_rg = any(in==origin(r)');
                    if ~any(org_rg)
                        error('region of origin not active');
                    end
                    r_and_similar = starting & (origin'==origin(r)); % region, and possibly other regions starting at same time from same region
                    x_same(r,t) = x_same(org_rg,t);
                    Wsame(r_and_similar,:,t) = Wsame(org_rg,:,t);
                    Wsame(:,r_and_similar,t) = Wsame(:,org_rg,t);
                else %starting from scratch
                    x_same(r,t) = x0all(r);
                    Wsame(r,r,t) = sigma0;
                end
            end
        end
        
        
        M_in = ARmatrix(Gxi, in, rho, dt); % auo-regressive matrix
        x_diff(in,t) = M_in*x_same(in,t) +I(in,t); %one-step prediction (eq 2.13)
        Wdiff(in,in,t) = M_in*Wsame(in,in,t)*M_in + dt*sigma*eye(length(in)); %one-step variance prediction (eq 2.14)
        
        aa = permute(A(in,t,:),[1 3 2]); % number of artifacts for each region at this time step
        if binary_obs
            asum =sum(aa,2); % total number of artifacts for each region at this time step
        else
            asum = cellfun(@length,aa);
        end
        if sum(asum)>0 % if there are observations at this time step
            if any(isnan(x_diff(in,t))) || any(abs(x_diff(in,t))>1e100)
                fprintf('nan/diverged values in predicted mean!\n');
                LLH = nan;
                x_tT = nan(R,T+1);
                W_tT = nan(R,R,T+1); % sigma^2_{t|T} (indices from 0 to K)
                Wt = nan(R,R,T+1);
                W_diff = nan(R,R,T);
                return;
            end
            
            % compute approximate trait distribution given artifacts up to t
            [x_same(in,t+1), Wsame(in,in,t+1), lh] = logregprior(x_diff(in,t), aa, Wdiff(in,in,t), options.observations, epsilon); % prior mean, observation, prior covariance
            LLH = LLH + log(lh); % for gaussian observations only
        else % if no observation, then posterior is simply prior
            x_same(in,t+1) = x_diff(in,t);
            Wsame(in,in,t+1) = Wdiff(in,in,t);
        end
        
    end
    
    % E-step II: backward fixed interval smoothing (FIS) algorithm
    x_tT = zeros(R,T+1); % x_{k|K}
    W_tT = zeros(R,R,T+1); % sigma^2_{t|T} (indices from 0 to K)
    x_tT(:,T+1) = x_same(:,T+1); % final conditions
    W_tT(:,:,T+1) = Wsame(:,:,T+1);
    W_diffT = zeros(R,R,T); % W_{t-1,t}|T (indices from 0 to K-1) (watning: the only sigma that is for not meant for its square value)
    for t=T:-1:1 % here we take from K to 1 and not K-1 to 0...
        
        active = onperiod(:,t); % regions active for that time step and subsequent one
        in = active; %find(active|starting|stopping); % index of active regions
        
        % add stopping points
        stopping = active & ~onperiod(:,t+1); % last active time point
        x_tT(stopping, t+1) = x_same(stopping,t+1);
        W_tT(stopping,:,t+1) = Wsame(stopping,:,t+1);
        W_tT(:,stopping,t+1) = Wsame(:,stopping,t+1);
        
        M_in = ARmatrix(Gxi, find(in), rho, dt); % auto-regressive matrix
        At = Wsame(in,in,t)*M_in/Wdiff(in,in,t); % eq 2.18 (also from S?derkvist 1995)
        x_tT(in,t) = x_same(in,t) + At*(x_tT(in,t+1)-x_diff(in,t)); % eq 2.17
        W_tT(in,in,t) = Wsame(in,in,t) + At*(W_tT(in,in,t+1)-Wdiff(in,in,t))*At'; % eq 2.19
        W_diffT(in,in,t) = At*W_tT(in,in,t+1); % (eq 2.20 with u=k+1)-> the formula in 2.20 applies to variance, not std, see DeJong & Kinnon
        
    end
    
    % E-step III: state-space covariance algorithm
    Wt = zeros(R,R,T+1);
    W_diff = zeros(R,R,T);
    for t=1:T+1
        in = onperiod(:,t);
        Wt(in,in,t) =  W_tT(in,in,t) + x_tT(in,t)*x_tT(in,t)'; % eq 2.22
        
        % cross-time step terms
        if t<T+1
            in_next = onperiod(:,t+1);
            W_diff(in,in_next,t) = W_diffT(in,in_next,t) + x_tT(in,t)*x_tT(in_next,t+1)'; % eq 2.21
            
        end
    end
    
    if ~strcmp(options.observations, 'continuous')
        
        % compute log-likelihood using Gauss-Hermite quadrature
        % (each time, we approximate log p(x_t | x_0..t-1) )
        n_gh = 20; % number of samples for Gauss-Hermite
        WW = zeros(R,T);
        for r=1:R
            WW(r,:) = Wdiff(r,r,1:T);
        end
        if any(WW(:)<0)
            %  warning('sqs');
        end
        A2 = reshape(A, R*T,1+binary_obs);
        LLH = LLH_gaussherm(reshape(x_diff(:,1:T),R*T,1),WW(:),A2,n_gh, options.observations);
        if isnan(LLH)
            2;
        end
    end
    
end
end

%%  M-step
function [par_hat, I,H_mstep] = Mstep(dt, R,T, x_tT, Wt, W_diff, E, G, onperiod, A, W_tT, options, origin)

with_ext = options.with_ext;
with_diffusion = options.diffusion;
fixedinitialspread = options.fixedinitialspread;

onperiod(:,end) = []; % remove final step

% update initial parameters
[starttime, stoptime] = startstoptime(onperiod); % get initial and last active time points for each region
sumvarini = 0;
xini = [];
Rini = sum(~origin); % number of initial regions (not inheriting cultural trait from other)

for r= find(~origin)
    xini(end+1) = x_tT(r,starttime(r));
    sumvarini = sumvarini + Wt(r,r,starttime(r));
end

if ~options.individualinitialpoint % one single parameter for initial mean
    x0 = sum(xini)/Rini; % mean initial value
    sigma0 = sumvarini/Rini - x0^2; % initial variance
else  % one parameter per de novo region
    x0 = xini;
    sigma0 =  (sumvarini- sum(x0.^2))/Rini;
end
if sigma0<0
    warning('negative initial noise!');
end

ss = @(x) sum(x(:)); % sum over all elements of matrix
tr = @(x) sum(diag(x)); % sum over diagonal elements of matrix (trace)
sumE = sum(E(onperiod)); %ss(E); % can be moved out of loop
ndatapoints = ss(onperiod(:,1:T)); %R*T

% update state transition parameters (rho, xi, lambda, gamma)
onperiod3 = permute(onperiod, [3 1 2]); % 1 x region x time
onperiod1 = permute(onperiod, [1 3 2]); % region x 1 x time

%% compute quadratic terms to estimate gamma, lambda, xi, rho
beta0 =  sum(bsxtimes(Wt(:,:,1:T), onperiod3,onperiod1),3);
beta1 = 0;
beta2 = 0;
Galpha = 0;
Gmu = 0;
sumGWdiff = 0;
for t=1:T
    onp = onperiod(:,t);
    G_in = G(onp,onp);
    Gtilde = G_in - diag(sum(G_in,2)); %connectivity matrix with negative terms along diagonal
    
    beta1 = beta1 +  ss( Wt(onp,onp,t).*Gtilde);
    beta2 = beta2 +  ss( bsxtimes( Wt(onp,onp,t), Gtilde^2));
    
    alpha = x_tT(onp,t) * E(onp,t)';
    Galpha = Galpha + ss(bsxtimes(Gtilde ,alpha));
    
    Gmu = Gmu + sum(Gtilde * x_tT(onp,t));
    
    sumGWdiff = sumGWdiff + ss(Gtilde.*W_diff(onp,onp,t)) ;
end

sxm = sum(bsxtimes(x_tT(:,1:T),onperiod),2); % sum_{t=1}^T x_{t-1|T}
sumWdiff = sum(bsxtimes(onperiod3,onperiod1 , W_diff),3);

% matrix for quadric term
C = zeros(4,4);
C(1,:) = [ndatapoints,        sumE,             Gmu,                    -sum(sxm)];
C(2,:) = [sumE,          ss(E.^2),            Galpha, -ss(bsxtimes(x_tT(onperiod),E(onperiod)))];
C(3,:) = [Gmu,          Galpha,               ss(beta2),         -beta1];
C(4,:) = [-sum(sxm),   -ss(bsxtimes(x_tT(onperiod),E(onperiod))), -beta1,         +tr(beta0)];

% vector for linear terms
D = zeros(4,1);
D(1) = 0; %sum(x_tT(:,T+1)-x_tT(:,1));
for r=1:R
    D(1) = D(1) + x_tT(r,stoptime(r)+1) - x_tT(r,starttime(r));
end
D(2) = ss(diff(x_tT,1,2).*E);
D(3) = sumGWdiff - beta1;
D(4) = tr(beta0) - tr(sumWdiff);

par_bin = true(1,4); % which parameters are free (by default all)
if ~with_diffusion % remove terms related to eta
    C(3,:) = [];
    C(:,3) = [];
    D(3) = [];
    par_bin(3) = false;
end
if ~with_ext % remove terms related to lambda
    C(2,:) = [];
    C(:,2) = [];
    D(2) = [];
    par_bin(2) = false;
end

try chol(C);
catch ME
    warning('Matrix is not symmetric positive definite')
end

% estimate [gamma, lambda, xi, rho] = inv(C)*D/dt
newpar = zeros(4,1);
newpar(par_bin) = C \ D /dt; % invert linear system

gamma = newpar(1);
lambda = newpar(2);
xi = newpar(3);
rho = newpar(4);

% update autoregressive matrix and overall input
I = dt*(lambda*E+gamma); % overall input onto each region
I(~onperiod) = 0;


%% estimate noise variance sigma2
sumWt = 0;
for r=1:R
    sumWt = sumWt + sum(Wt(r,r,starttime(r)+1:stoptime(r)+1));
end

beta0M2 = 0;
sumMWdiff = 0;
sumMxI = 0;
for t=1:T
    onp = onperiod(:,t);
    M_in = ARmatrix(G*xi, find(onp), rho, dt); % auo-regressive matrix
    beta0M2 = beta0M2 +  ss( Wt(onp,onp,t).* M_in^2);
    sumMWdiff = sumMWdiff + ss(M_in.*W_diff(onp,onp,t)) ;
    sumMxI = sumMxI + x_tT(onp,t)'*M_in*I(onp,t);
end
SS = sumWt + beta0M2 + ss(I.^2) -2*sumMWdiff	...
    - 2*ss(x_tT(:,2:T+1).*I) +2*sumMxI;
if SS<0
    warning('negative noise!');
end

sigma = SS/(ndatapoints*dt);


%% group parameters
par_hat = [newpar' sigma x0 sigma0];

if ~with_diffusion
    par_hat(3) = []; % remove xi
end
if ~with_ext % remove terms related to lambda
    par_hat(2) = [];
end
if fixedinitialspread
    par_hat(end) = [];
end

%% estimate observation noise for continuous valued artifacts
continuous_obs = strcmpi(options.observations, 'continuous');
if continuous_obs
    epsilon = 0;
    nobs_tot = 0;
    for t=1:T
        yy = A(:,t);
        nobs = cellfun(@length, yy);
        nobs_tot = nobs_tot + sum(nobs);
        yy = [yy{:}];
        idx = repelem(1:R, nobs);
        this_eps =  tr(Wt(idx,idx,t)) + sum(yy.^2) - 2*sum(yy.*x_tT(idx,t)');
        if this_eps<0
            error('!!!');
        end
        epsilon = epsilon + this_eps;
    end
    epsilon = epsilon/nobs_tot;
    par_hat(end+1) = epsilon;
end

%% compute Hessian for Q
if nargout>2
    nextra = with_ext+with_diffusion;
    npar = options.npar;
    H_mstep = zeros(npar);
    H_mstep(1:2+nextra,1:2+nextra) = -dt*C/sigma; % bilinear form
    H_mstep(3+nextra,3+nextra) = -ndatapoints/(2*sigma^2); % hessian w.r.t sigma
    if options.individualinitialpoint
        idx = 3+nextra+(1:Rini);
        H_mstep(idx,idx) = -eye(Rini)/sigma0; % hessian w.r.t x0
    else
        H_mstep(4+nextra,4+nextra) = - R/sigma0;  % hessian w.r.t x0
    end
    if ~fixedinitialspread
        cc = npar-continuous_obs;
        H_mstep(cc,cc) = - R/(2*sigma0^2);
    end
    if continuous_obs
        H_mstep(npar,npar) = -nobs_tot/(2*epsilon^2);
    end
end

end


%% logistic/Poisson regression with prior (taken from Rasmussen & Williams, algorithm 3.1)
function [xx, cov, lh] = logregprior(mm, aa, K, obs, epsilon)
oldlogpost = -Inf;
notconverged = 1;
xx = mm; %initialize value
%  yy = permute(Y(in,t,:),[1 3 2]); % number of non-romantic (first col) and romantic (sec col) books for each region at this time step
if strcmpi(obs,'binary')
    nY = sum(aa,2); % total number of artifacts for each region at this time step
else
    sY = cellfun(@sum, aa); % sum of counts for all artifacts in each region
    nY = cellfun(@length,aa); % number of artifacts in each region
end
lh = nan;

% gaussian observations, use analytical solution
if strcmpi(obs, 'continuous')
    R = length(mm);
    idx = repelem(1:R, nY);
    nobs = sum(nY);
    
    C = eye(R);
    C = C(idx,:); % projection matrix (one line per observation)
    aa = [aa{:}]';
    obs_cov = epsilon*eye(nobs); % covariance matrix of noise process
    Kidx = K(idx,idx);
    lh = mvnpdf(aa, mm(idx), Kidx + obs_cov); % likelihood of data
    
    KGM = K(:,idx)/(Kidx + obs_cov); % Kalman gain matrix
    
    xx = mm + KGM*(aa-xx(idx)); % new mean
    cov = (eye(R)-KGM*C)*K;
    return;
end

% iterate until convergence
ii =0;
while notconverged
    %update parameters using IRLS
    
    if strcmpi(obs,'binary')
        Phi = 1 ./ (1 + exp(-xx)); % predicted observation probability
        llh = aa(:,1)'*log(1-Phi) + aa(:,2)'*log(Phi);
    else % count
        Phi = exp(xx); % predicted rate
        llh = sY'*xx - nY'*Phi; % LLH for Poisson observations
    end
    dx = xx - mm;
    logpost = -dx' / K*dx/2 + llh; %log posteior
    
    cc = 0;
    while (ii>0) && (logpost<oldlogpost-1e-12) && sqrt(sum((xx-xold).^2))>1e-12 % in case the log-posterior decreases after an iteration, which shouldn't happen
        xx = (xx+xold)/2;
        if strcmpi(obs,'binary')
            Phi = 1 ./ (1 + exp(-xx)); % predicted observation probability
            llh = aa(:,1)'*log(1-Phi) + aa(:,2)'*log(Phi);
        else
            Phi = exp(xx); % predicted rate
            llh = sY'*xx - nY'*Phi; % LLH for Poisson observations
        end
        dx = xx - mm;
        logpost = -dx' / K*dx/2 + llh; %log posteior
        cc = cc +1;
        if cc==20
            disp('?');
        end
    end
    
    if strcmpi(obs,'binary')
        grad = aa(:,2) - nY.*Phi; % gradient of likelihood (eq 3.15)
        W = nY.* Phi .* (1-Phi);
    else
        grad = sY - nY.*Phi;
        W = nY.*Phi;
    end
    xold = xx;
    
    sqW = diag(sqrt(W));
    B = eye(length(W)) + sqW*K*sqW;  % hessian wr.t. predictor
    try
        L = chol(B);
    catch
        warning('non definite positive');
        xx = nan(size(xx));
        cov = nan(length(xx));
        return;
    end
    
    b =  grad + xx.*W ;
    a = b - sqW * (L \ ( L'\(sqW * K *b)));
    c = mm - K * sqW * (L \ ( L'\(sqW * mm))); % equation is changed slightly due to non-zero prior mean
    if any(isnan(a))
        error('w');
    end
    xx = K*a + c;
    
    notconverged = abs(logpost-oldlogpost)>1e-6;
    oldlogpost = logpost;
    ii = ii+1;
    if ii>20 % sometimes looks like it's trapped in oscillating between two values, let's try to help him out
        xx = (xx + xold)/2;
    end
    if ii==100
        fprintf('!');
    end
end

% covariance
cov = K - K * sqW * (L' \ ( L\(sqW * K)));
cov = (cov+cov')/2; % enforce symmetry if not the case due numerical problems


end


%% get initial and last active time points for each region
function [starttime, stoptime] = startstoptime(onperiod)
R = size(onperiod,1);
starttime = zeros(1,R);
stoptime = zeros(1,R);
for r=1:R
    starttime(r) = find( onperiod(r,:),1); % initial point for each region
    stoptime(r) = find( onperiod(r,:),1,'last'); % initial point for each region
end
end


%% get likelihood of binomial observations given gaussian latent posterior
function LLH = LLH_gaussherm(M,S,A,n_gh, obs)
[x_gh, w_gh] = GaussHermite(n_gh); % draw samples and weights from Gauss-Hermite
w = w_gh/sqrt(pi);
ndatapoint = length(M);
LLH = 0;
for o=1:ndatapoint
    aa = A(o,:);
    if strcmpi(obs,'binary')
        nobs = sum(aa);
    else
        nobs = length(aa{1});
    end
    xx =  sqrt(2*S(o))*x_gh + M(o); % range of sample values
    if nobs>0 % if any observation for the corresponding data point
        if strcmpi(obs,'binary')
            LL = logistic(xx); % logistic function for all weights
            Lmarg = w' * LL; % marginalized over value of x
            llh = [aa(1)*log(1-Lmarg)  aa(2)*log(Lmarg)];
            llh(aa==0) = 0;
            LLH = LLH + sum(llh);
            if any(~isreal(sum(llh)))
                warning('negative likelihood');
            end
        else
            ncount = sum(aa{1}); % total number of observations
            LL = ncount*xx - nobs*exp(xx);
            Lmarg = w' * LL; % marginalized over value of x
            LLH = LLH + Lmarg;
        end
        
    end
end

LLH = real(LLH);

end

%% logistic ("inverse logit") function:
function p = logistic(x)
p = exp(x)./(1+exp(x));
p(x > 1e2) = 1;
p(isinf(x) & x < 0) = 0;
end


%% decompose parameters
function [gamma,lambda,xi,rho,sigma,x0,sigma0, epsilon] = par_decomp(pars,with_ext, options) %with_diff, fixedinitialspread)
gamma = pars(1); % bias in state update equation
cc = 2;
if with_ext
    lambda = pars(cc); %dependence on stimulus
    cc = cc+1;
else
    lambda = 0;
end
if options.diffusion
    xi = pars(cc); % dependence on neighbouring regions
    cc = cc+1;
else
    xi = 0;
end
%nextra = with_ext + options.diffusion;
rho = pars(cc); %leak parameter
sigma = pars(cc+1); % variance of auto-regressive noise (! here sigma stands for sigma^2 in equations)
x0 = pars(cc+1+(1:options.n_x0)); % initial value of latent variable
cc = cc+2+options.n_x0;
if options.fixedinitialspread
    sigma0 = 1; % std of initial point
else
    sigma0 = pars(cc);
    cc = cc+1;
end
if strcmpi(options.observations, 'continuous')
    epsilon = pars(cc); % ariance of observation noise (gaussian)
    cc = cc+1;
else
    epsilon = nan;
end
assert(length(pars) == cc-1);

end

% transform vectors of subset of observable into number of
% observables for each region and time
function   Asubset = observation_subset(A, subset)
if iscell(A)
    Acount = cellfun(@length,A); % counts
elseif  ndims(A)==3 % binary
    Acount = A;
else % direct
    Acount = 1*~isnan(A);
end
nA = sum(Acount(:)); % total number of artifacts

assert(all(subset<=nA)); % make sure indices for observations are not too large
subset = sort(subset);

Yc = [0 cumsum(Acount(:))']; % indices for each region, time

if iscell(A) % count
    Asubset = cell(size(A));
    for i=1:numel(A)
        this_subset = (subset>Yc(i)) & (subset<=Yc(i+1)); % observations that we include for this time and region
        this_ind = subset(this_subset)-Yc(i); % indices of selected observations for this time and region
        Asubset{i} = A{i}(this_ind);
    end
elseif  ndims(A)==3  %binary
    Asubset = zeros(size(A));
    for i=1:numel(A)
        Asubset(i) = sum( (subset>Yc(i)) & (subset<=Yc(i+1)) );
    end
    
else % direct
    Asubset = nan(size(A));
    for i=1:numel(A)
        if any(subset==Yc(i))
            Asubset(i) = A(subset==Yc(i));
        end
    end
end
end

function C = bsxtimes(varargin)
if nargin>2
    C = bsxtimes(bsxtimes(varargin{1},varargin{2}),varargin{3:end});
else
    C = bsxfun(@times, varargin{1},varargin{2});
end
end

% recode function repelem for versions earlier than 2015
function y = repelem(v,n)
cs = [0;cumsum(n(:))];
y = zeros(cs(end),1);
for i=1:length(n)
    idx = cs(i)+1:cs(i+1);
    y(idx) = v(i);
end
end

%% compute influence of vertical, horizontal transmission and evoked culture
function infl = influence_analysis(T, Tgeneration, gamma,lambda,xi,rho,sigma, E, G, onperiod, x_tT)
ndatapoints = sum(sum(onperiod(:,1:end-1)));

x_ev = nan(1, ndatapoints);
x_horz = nan(1, ndatapoints);
v_noise = nan(1, ndatapoints);
i=0;
e_rho = exp(-rho*Tgeneration);

for t=1:T
    on = onperiod(:,t);
    n_active = sum(on);
    idx = i+ (1:sum(on));
    
    G_in = G(on,on);
    Gtilde = G_in - diag(sum(G_in,2)); %connectivity matrix with negative terms along diagonal
    M_in = rho*eye(n_active) - xi*Gtilde; % dx/dt = - Mx  + ...
    eM = expm(-M_in*Tgeneration);
    
    x_ev(idx) =  lambda * ( eye(n_active) - eM) * E(on,t); % evoked factor
    x_horz(idx) = (eM - e_rho *eye(n_active))*(x_tT(on,t)-gamma); % horizontal tranmission)
    
    v_noise(idx) = sigma/2* diag(M_in \ (eye(n_active)-eM^2));
    
    i = i+n_active;
end

% summed variance from vertical transmission, horizontal, evoked and noise
V(1) = exp(-2*rho*Tgeneration)*var(x_tT(onperiod)); % variance of vertical transmission term
V(2) = var(x_horz);
V(3) = var(x_ev);
V(4) = mean(v_noise);


sumV = sum(V); % sum of variance
infl = V(1:3)/sumV;% proportion of total cultural trait variance  ( vertical transmission, horizontal, evoked)


end


%% compute auto-regressive matrix
function         M_in = ARmatrix(Gxi, in, rho, dt)
Gxiin = Gxi(in,in);
Gtilde = Gxiin - diag(sum(Gxiin,2)); %connectivity matrix with negative terms along diagonal
M_in = (1-rho*dt)*eye(length(in)) + dt*Gtilde; % auto-regressive matrix
end
