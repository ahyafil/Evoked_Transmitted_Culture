function [x0, B, lambda, xi, C, llh, exitflag, output] = etc_fit_binary(dt, E, G, A, par_ini, options, origin)
% [x0, B, lambda, xi, C, llh, exitflag, output] = etc_fit_binary(dt, E, G, A, pars0, options, origin)
% fits Evoked and Transmitted Culture (ETC) models with binary cultural
% traits and binary artifacts.
%
% INPUT ARGUMENTS:
% - dt: time step (scalar or vector of length T)
% - E: ecological variable in each region and each time step (R x T matrix,
% where R is the number of regions and T the number of time steps). Use
% zeros(R,nT,0) if no ecological variable
% - G: connectivity matrix for cultural diffusion (square matrix of size R)
% - A: binary artifacts data: R x T x 2 matrix with numbers of artifacts for
%      artifacts of value 0 in A(:,:,1) and artifacts of value 1 in
%      A(:,:,2)
%
% - par_ini: initial value of parameters (vector for [x0, B, lambda, xi, C]). 
% Use [] for default initial value
% 
% - options: structure with options for fitting. Possible fields are:
%        - 'maxiter': maximum number of iterations for EM algorithm (default: 1000)
%        - 'TolFun': tolerance value for stopping EM algorithm (default
%        1e-4)
%        - 'initialpoints': number of initial sets of parameter values for EM algorithm (default: 50)
%        - 'bootstrap': number of parametric boostraps used to compute standard errors over parameters (default:0).
%        or two parameters (one for each transition)
%        - 'verbose': boolean sets verbosity of algorithm (true by default)
%     
% - origin: region of origin for each region (vector of length R, use 0 for
% regions created de novo)
%
% OUTPUT PARAMETERS:
%
%  - x0: probability of up-value for initial value of cultural trait (regions
% created de novo)
% - b: vector of transition [b_up b_down]
% - lambda: susceptibility to ecological factor
% - xi: cultural diffusion parameter
% - C: cultural artifacts parameter (vector of two element: probability of
% artifact of value 1 for T=down, probability of artifact of value 1 for
% T:up)
%
% llh: log-likelihood for estimated parameters
% exitflag: should be positive if algorithm converged correctly
% output: structure with a lot of fields (most should be self-explanatory)
%
% See also etc_rnd, etc_test, etc_fit_binary


%% process input

% default parameters
initialpoints = 20; % number of initial points in the minimization procedure
maxiter = 1000;
TolFun = .0001;
lambda_sym = 1; % whether we use just one parameter for lambda (with opposite symmetric effect of external variable on transition from down- to up-state, or 2)
verbose = 1;
bootstrap = 100;


if nargin < 6
    options = struct;
end

[R, T,~] = size(E);  %number of regions x number of time steps x n external factors

if nargin<7 % default: all regions created from scratch
    origin = zeros(1,nregion);
end

% period for each region
if isfield(options, 'onperiod')
    onperiod = options.onperiod;
else
    onperiod = ~any(isnan(E),3);
end
if any(~any(onperiod))
    error('there should not be time point with no active dataset');
end
ndatapoint = sum(onperiod(:)); % total number of time points
A_on = reshape(A,R*T,2); % merge all observations across regions (for M-step)
A_on = A_on(onperiod,:); % observations points only for active regions
nA = sum(A_on(:)); % total number of artifacts


if isfield(options, 'maxiter') % maximum iterations of the EM algo
    maxiter = options.maxiter;
end
if isfield(options, 'TolFun')
    TolFun = options.TolFun;
end
if isfield(options, 'initialpoints')
    initialpoints = options.initialpoints;
end
if isfield(options, 'boostrap')
    bootstrap = options.bootstrap;
end
if isfield(options, 'symmetriclambda')
    lambda_sym = options.symmetriclambda;
end
if isfield(options, 'verbose')
    verbose = options.verbose;
end

with_ext = ~isempty(E); % whether there is any external variable
n_lambdapar = with_ext*(2- lambda_sym); % number of parameters lambda for influence of external variable
npar = 6+n_lambdapar;

% index of parameters
idx.b = [1 2]; % spontaneous transition rate
idx.lambda = 3:2+n_lambdapar; % variable-dependent transition rate
idx.xi = 3+n_lambdapar;
idx.C = [4 5]+n_lambdapar; % probability of romantic book associated with down- and up-state
idx.x0 = 6+n_lambdapar; % initial value for hidden variable


if with_ext
    E_notlast = E(:,1:end-1); % exclude last data point from each region from E (not used for M-step)
    Espan = max(E(:))-min(E(:));
end
dt_notlast = dt(1:end-1);

% map 2^R steps
Rmax = max(sum(onperiod(:,1:end-1) | onperiod(:,2:end)));
SB = false(2^Rmax,R); % value for in region for each multiregion state
kk = 0:2^Rmax-1;
for i=1:R
    ll = mod(kk,2);
    SB(:,i) = ll;
    kk = (kk-ll)/2;
end

dt_max = max(dt); % maximum time interval

%% default values if not provided
if nargin < 4 || isempty(par_ini)
    par_ini(idx.x0) = .5;
    par_ini(idx.b) =  .1*[1 1]/dt_max;
    par_ini(idx.lambda) = 0;
    par_ini(idx.xi) = 0;
    par_ini(idx.C) = [.25 .75];
end


pars_all = nan(initialpoints,length(par_ini));
llh_all = zeros(1,initialpoints);
exitflag_all = zeros(1,initialpoints);

%% run optimization, each time with a distinct initial point
for init = 1:initialpoints
    if initialpoints>1
        fprintf('starting point #%d/%d\n',init,initialpoints);
    end
    
    % initial values for parameters
    if init==1 % for the first onem always use the provided initial point
        par_hat = par_ini; 
        B = par_hat(idx.b);
        lambda = par_hat(idx.lambda);
        xi = par_hat(idx.xi);
        C = par_hat(idx.C);
        x0 = par_hat(idx.x0);
        
    else % for the following ones, compute randomly
        x0 = rand;
        if with_ext
            lambda_max = 1/10/dt_max/Espan;
            lambda = randn(1,2)*lambda_max/4; %making sure we dont have 0 or 1s in transition matrix
            lambda(abs(lambda)>lambda_max) = lambda_max*sign(lambda(abs(lambda)>lambda_max)); % abs value limited to 4
            if lambda_sym
                lambda(2) = [];
            end
            if lambda_sym
                lbd = [lambda -lambda];
            else
                lbd = lambda;
            end
            
                     if lambda_sym
                brange = [ -min(E(:)*lambda) ; .1/max(dt)-max(E(:)*lambda)]; % range values for b
            else
                error('not coded');
            end
            B = brange(1,:) + rand(1,2).*diff(brange);
            
            xi = abs(randn);
            xi(abs(xi)>1) = 1*sign(xi(abs(xi)>1)); % abs value limited to 1
            xi = .1*xi/max(sum(G))*min(min(bsxfun(@plus,E(:)*lbd,B)));
        else
            B = rand(1,2)/10/dt_max;
            
            xi = abs(randn);
            xi(abs(xi)>1) = 1*sign(xi(abs(xi)>1)); % abs value limited to 1
            xi = .1*xi/max(sum(G))*min(B);
        end
        C = sort(rand(1,2));
        
    end
    
    [lbd, with_ext, lambda_sym] = lambdapar(lambda); % get from lambda parameter to values of state-dependent transition for both types of transition
    if with_ext && any(any(bsxfun(@plus,E(:)*lbd,B)<0))
        warning('negative rate');
    end
      
    
    %% run optimization procedure
    [C,x0,B,lambda,xi,exitflag_Mtrans, llh] = EMalgo(...
        R, T,E, dt, A, G, B, lambda, x0, xi, C, SB,onperiod, origin, with_ext, lambda_sym, dt_notlast, TolFun, maxiter, A_on,idx,npar, verbose);

   
    % symmetry in parameter space: make sure first state corresponds to
    % lower proba of observable
    if C(1)>C(2) % if it's the converse, just invert it
        C = C([2 1]);
        B = B([2 1]);
        if with_ext
            if lambda_sym
                lambda = -lambda;
            else
                lambda = lambda([2 1]);
            end
        end
        x0 = 1 - x0;
    end
    
    par_hat(idx.C) = C;
    par_hat(idx.b) = B;
    par_hat(idx.lambda) = lambda;
    par_hat(idx.x0) = x0;
    par_hat(idx.xi) = xi;
    
    
    exitflag = exitflag_Mtrans;
    output = struct; % !!! should fill that
    pars_all(init,:) = par_hat;
    llh_all(init) = llh;
    exitflag_all(init) = exitflag;
    output_all(init) = output;
    
end

%% select iteration that yield highest LLH
[llh, besti] = max(llh_all);
par_hat = pars_all(besti,:);
B = par_hat(idx.b);
lambda = par_hat(idx.lambda);
C = par_hat(idx.C);
x0 = par_hat(idx.x0);
xi = par_hat(idx.xi);
exitflag = exitflag_all(besti);
output = output_all(besti);

% compute again gamma and eta
if lambda_sym
    lbd = [lambda -lambda];
else
    lbd = lambda;
end
[llh, gamma, eta] = Estep(R, T, E, dt, A, G, B, lambda, x0, xi, C,SB,onperiod, origin);

output.gamma = gamma;
output.gamma_marg = gamma_marg(gamma, SB, onperiod); % marginalize (over other states) to get probability of up-state for each region and each time step
output.eta = eta;


%% compute hessian
fprintf('computing hessian ...');
[~,~,~,~,~,~,~,~,hess_Mstep] = Mstep(C,x0,B,lambda,xi,  gamma, eta, R, E, G, dt_notlast,SB,A_on,TolFun,idx,npar,onperiod, origin); % first get hessian from Mstep
%try

%then compute dM/dpar using finite difference
MM = nan(npar);
for p=0:npar % first one is one more E and M step, further ones are E+M with infinitesimal change along one dim
  dx = 1e-3; 
  par_dx = par_hat + dx*((1:npar)==p); % change one parameter by dx
    
    % now Estep and Mstep to compute change in parameters
    try
        [chk(p+1), gamma_dx, eta_dx] = Estep(R, T, E, dt, A, G, par_dx(idx.b), par_dx(idx.lambda), par_dx(idx.x0), par_dx(idx.xi), par_dx(idx.C),SB,onperiod, origin);
        [C_dx,x0_dx,b_dx,lambda_dx,xi_dx] = Mstep(...
            par_dx(idx.C),par_dx(idx.x0),par_dx(idx.b),par_dx(idx.lambda),par_dx(idx.xi),  gamma_dx, eta_dx, R, E, G, dt_notlast,SB,A_on,TolFun,idx,npar,onperiod, origin);
        par_dx2(idx.C) = C_dx;
        par_dx2(idx.x0) = x0_dx;
        par_dx2(idx.b) = b_dx;
        par_dx2(idx.lambda) = lambda_dx;
        par_dx2(idx.xi) = xi_dx;
        if p==0
            par_hat_plusone = par_dx2;
        else
            MM(p,:) = (par_dx2 - par_hat_plusone)/dx;
        end
    catch
        warning('error while computing hessian w.r.t %d parameter',p);
    end
end

hess = hess_Mstep .* (eye(npar)-MM); % Hessian in EM, see. e.g. Jamshidian & Jennrich 2000, equation 1
output.hess = hess;
fprintf('\n');

output.covb = inv(-hess); %Covariance matrix

% standard error of estimates
output.se = sqrt(diag(output.covb))';

% T-statistic for the weights
output.T = par_hat ./ output.se;

% p-value for significance of each coefficient (two-sided Wald test)
output.p = 2*normcdf(-abs(output.T));


% compute BIC and AIC
output.BIC = llh - npar/2*log(nA); % Bayes Information Criterion
output.AIC = 2*npar - 2*llh; % Akaike Information Criterior
output.AICc = output.AIC + 2*npar*(npar+1)/(nA-npar-1); % AIC corrected for sample size
output.logevidence = llh + npar*log(2*pi)/2 - log(det(hess))/2; % % from Laplace approximation (Bishop eq 4.137)
output.n_init = initialpoints;
output.LLH_all = llh_all;
output.pars_all = pars_all;

% compute how many initial points arrived at the (overall) minimum
% objective function value
Fvaldiff = bsxfun(@minus,llh, llh_all); % difference between final point for each starting point and overall final point
output.MinFvalCount = sum(Fvaldiff<TolFun);

output.ndatapoint = ndatapoint;
output.nobs = nA;

 %% parametric bootstrapping to compute uncertainty over parameters (Visser et al 2000)
if bootstrap>0
    
    par_bts = zeros(bootstrap, npar);
    
 
    fprintf('RUNNING BOOTSTRAPPING');
    
    parfor b=1:bootstrap % for each permutation
        
        fprintf('.');
        
        % generate sample from best-fitting values
               nA_timepoint = sum(A,3);
       Ybs = etc_rnd_binary(dt, E, G, nA_timepoint, x0, B, lambda, xi, C, origin );
            
        % compute best-fitting parameters for this values
           [C_bts,x0_bts,A_bts,lambda_bts,xi_bts] = EMalgo(...
        R, T,E, dt, Ybs, G, B, lambda, x0, xi, C, SB,onperiod, origin, with_ext, lambda_sym, dt_notlast, TolFun, maxiter, A_on,idx,npar, verbose);

        
        par_bts(b,:) = [C_bts,x0_bts,A_bts,lambda_bts,xi_bts];
        fprintf('*');
    end
    fprintf('done\n');
   
    output.par_bts = par_bts;
    output.mean_bts = mean(output.par_bts,1);
    oneside = min( mean(output.par_bts>=0), mean(output.par_bts<=0)); % smaller tail for bottstrap samples
    output.p_bts = min(2*oneside,1); % p-values from bts
    output.se_bts = std(output.par_bts,1);
else
    output.par_bts = [];
    output.mean_bts = [];
    output.p_bts = [];
    output.se_bts = [];
    output.output_bts = [];
end

end

%% compute M step
function  [C,x0,B,lambda,xi,new_F,old_F,exitflag_Mtrans, hess] = Mstep(C,x0,B,lambda,xi,  gamma, eta, nreg, E, G, dt_notlast,SB,A,TolFun, idx,npar,onperiod, origin)

gamma_m = gamma_marg(gamma, SB, onperiod); % marginalize (over other states) to get probability of up-state for each region and each time step

ZZ_ini = [nreg 0]; % summing responsabilities for down- and up-state over region for first time step
starttime = startstoptime(onperiod); % get initial and last active time points for each region
for r=1:nreg
    ZZ_ini = ZZ_ini + [-1 1]*gamma_m(r,starttime(r));
end

gamma_m = gamma_m(onperiod)';%
ZZ_C = [1-gamma_m; gamma_m] * A;  % summing responsabilities for down- and up-state over region and time

% Fm before maximization
old_Lemission = xlogy(ZZ_C, [1-C'  C'], 0);
old_Lini =  xlogy( ZZ_ini, [1-x0  x0],0);
old_Ltrans = - negQtransition([B lambda xi], eta, E, G, dt_notlast,SB,onperiod,x0, origin);
old_F = [old_Lemission  old_Lini old_Ltrans];

% update parameters for transition rate
[B, lambda, xi, Ftrans, exitflag_Mtrans,hess_trans] = transition_Mstep(eta, E, G, B, lambda, xi, dt_notlast, TolFun,SB,onperiod,x0, origin);

%emission probabilities
C = ZZ_C(:,2)' ./ sum(ZZ_C,2)'; % adapted from bishop 13.23
CC = [1-C'  C']; % probability for no-romantic and for romantic
Femission = xlogy(ZZ_C, CC, 0);
hess_emission = - sum(ZZ_C.*CC.^-2,2);
hess_emission = diag(hess_emission);

%update parameter for starting point
x0 = ZZ_ini(2) / sum(ZZ_ini); % vector of 2 (ini proba for down and up-state)
Fini = xlogy( ZZ_ini, [1-x0 x0],0);
hess_ini = - sum(ZZ_ini .*  [1-x0 x0].^-2);

% hessian
if nargout>8
    hess = zeros(npar);
    hess(idx.x0,idx.x0) = hess_ini;
    hess(idx.C, idx.C )= hess_emission;
    idx_b_lambda_xi = [idx.b idx.lambda idx.xi]; % indices for transition parameters
    hess( idx_b_lambda_xi,idx_b_lambda_xi) = hess_trans;
    
    % transition parameters also depend on x0, use finite difference
    dx = 1e-6;
    x0_dx = x0 + dx;
    [B_dx, lambda_dx, xi_dx] = transition_Mstep(eta, E, G, B, lambda, xi, dt_notlast, TolFun,SB,onperiod,x0_dx, origin);
    hess( idx_b_lambda_xi,idx.x0) = [B_dx-B lambda_dx-lambda xi_dx-xi]/dx;
    hess(idx.x0,idx_b_lambda_xi) = hess( [idx.A idx.lambda idx.xi],idx.x0);
end

new_F = [Femission Fini Ftrans];
if any(isnan(new_F))
    warning('nannnns');
end

end



%% fit parameter for transition of hidden variable
function [B, lambda, xi, Qtrans, exitflag,hessian] = transition_Mstep(eta, E, G, B, lambda, xi, dt, TolFun,SB,onperiod,x0, origin)

maxiter = 40;

usefmincon = 1; % whether to use fmincon or Newton update for minimization

% set inequality constraints on parameter space
Min = zeros(0,3+length(lambda)); % Min*x <= Nin
Nin = zeros(0,1);
MinE = min(E(:));
MaxE = max(E(:));
switch length(lambda)
    case 0 % no environmental variable
        %   lb = [0 0]; % each fixed rate must be positive (dont use this because it worsenss optimization)
    case 1 % symmetric parameter for env variable
        Min = -[1 0 MinE 0; 1 0 MaxE 0;0 1 -MinE 0; 0 1 -MaxE 0];
        Nin = [0 0 0 0]';
    case 2 % non-symmetric parameter for env variable
        Min = -[1 0 MinE 0 0; 1 0 MaxE 0 0; 0 1 0 MinE 0; 0 1 0 MaxE 0];
        Nin = [0 0 0 0]';
end

if usefmincon %gradient-based
    algo = 'trust-region-reflective'; 'interior-point';
    hessianfcn = 'objective'; % use objective Hessian computed in function
    
    optoptions = optimoptions(@fmincon, 'Algorithm', algo, 'GradObj','on', 'Display', 'off',...
        'TolCon',0, 'TolFun', TolFun, 'Hessian','off', 'DerivativeCheck', 'off', 'HessianFcn',hessianfcn,'MaxIter',maxiter);
    
    try
        [TP, negQtrans, exitflag,~,~,~,hessian] = fmincon(@(x) negQtransition(x, eta, E, G, dt,SB,onperiod,x0, origin), [B lambda xi], [], [], [],[],[],[],[], optoptions);
    catch ermsg
        rethrow(ermsg);
    end
    
    
    negQ_init = negQtransition([B lambda xi], eta, E, G, dt,SB,onperiod,x0, origin); % initial value
    
else %% Newton update, using hessian
    TP = [B lambda xi];
    it = 0;
    while it<maxiter && (it<2 || dQ>TolFun)
        
        [negQtrans, grad, hessian] = negQtransition(TP, eta, E, G, dt,SB,onperiod,x0, origin); % initial value
        if it==0 % initial point
            negQ_init = negQtrans;
            if isnan(negQtrans) &&  isnan(negQtransition(zeros(1,3+length(lambda)), eta, E, G, dt,SB,onperiod,x0, origin))
                error('nan initial value for transition free energy');
            end
            while isnan(negQtrans) %% if transitions is too large with respect to time bin, will create nan values
                TP = TP/2; % so we reduce initial values until we reach a non-nan value for transition
                [negQtrans, grad, hessian] = negQtransition(TP, eta, E, G, dt,SB,onperiod,x0, origin); % initial value
            end
            dQ = 0;
        else
            dQ =  prevnegQ - negQtrans; % improvement of Q over last step
        end
        %         fprintf('%d, %f, %f\n',it, negQtrans, dQ);
        if it==0 || dQ>0
            %Newton-Raphson update on parameters
            dTP = grad/hessian;
            TP = TP - dTP;
            prevnegQ = negQtrans;
            
        else % if we're doing worse than previous step, go back half the previous step
            dTP = dTP/2;
            TP = TP + dTP;
            dQ = Inf;
        end
        
        it = it+1;
    end
    
    negQtrans = prevnegQ; % best value along iteration
    if isinf(dQ)
        TP = TP+dTP;
    end
    if isnan(negQtrans) %&& ~isnan(negQ_init)
        warning('nan Mtrans');
    end
    
    %  disp('end of this max');
    exitflag = (dQ<=TolFun) && ~isnan(negQ_init);
end

if (negQtrans>negQ_init) || any(Min*TP'>Nin) % if fmincon landed on point worse than initial point or point that does not satisfy constraint
    warning('maximization worse than initial point');
    negQtrans = negQ_init;
    exitflag = -7;
else
    B = TP(1:2);
    lambda = TP(3:end-1);
    xi = TP(end);
end
Qtrans = -negQtrans;
hessian = -hessian;
end

%% E-step: compute alphas, betas, gamma, eta and LLH
function  [llh, gamma, eta] = Estep(nreg, ntime, E, dt, A, G, B, lambda, x0, xi, C,SB,onperiod, origin)

Rmax = max(sum(onperiod(:,1:end-1) | onperiod(:,2:end))); % maximum number of regions active at a time
nstate_max = 2^Rmax;
nTS = nreg+1; % number of possible transition state for each state
alpha = zeros(nstate_max,ntime);
beta = zeros(nstate_max,ntime);
alpha_normlog = zeros(1,ntime); % log_normalization for numeric purpose
beta_normlog = zeros(1,ntime);
pobs_normlog = zeros(1,ntime);

% pseudo-sparse transition matrix: map change in regions to new state
TM = zeros(Rmax+1,nstate_max); % matrix maps transition x oldstate to newstate, but also transition x newstate to oldstate
TM(1,:) = 1:nstate_max; % first column represents no-change
for i=1:Rmax % for each region
    upstate = SB(:,i);
    TM(i+1,upstate) = find(~upstate); % transition from up- to down-state
    TM(i+1,~upstate) = find(upstate);  % transition from down- to up-state
end

% compute transition matrix for each time step
AA = p_trans(B, lambda, xi, E, G, dt, SB,[],onperiod, x0, origin);

% probability of observations given hidden state
pobs = ones(nstate_max, ntime);
for t=1:ntime
    if t<ntime
        inreg = find(onperiod(:,t) | onperiod(:,t+1)); % regions active for that time step and subsequent one
    else
        inreg = find(onperiod(:,t)); % regions active for that time step and subsequent one
    end
    
    inreg_all{t} = inreg;
    nr(t) = length(inreg);
    ns(t) = 2^nr(t); % number of associated states
    SBsub = SB(1:ns(t),1:nr(t));
    log_pobs = zeros(1,ns(t));
    for r=find(onperiod(inreg,t))' %1:nreg
        r2 = inreg(r);
        log_pobs(SBsub(:,r)) = log_pobs(SBsub(:,r)) + log(1-C(2)).*A(r2,t,1) + log(C(2)).*A(r2,t,2); % log p(y_{r,t}|x_t)
        log_pobs(~SBsub(:,r)) = log_pobs(~SBsub(:,r)) + log(1-C(1)).*A(r2,t,1) + log(C(1)).*A(r2,t,2); % log p(y_{r,t}|x_t)
    end
    pobs_normlog(t) = - max(log_pobs); % we take the maximum over all values of hidden, to make sure there are all non 0s
    pobs(1:ns(t),t) =  exp(log_pobs + pobs_normlog(t)); %(1-C).^Y{r}(1,t) .* C.^Y{r}(2,t); % p(y_t|x_t) / max_y  p(y_t|x_t)
end

%    initialize alphas and betas
nr0 = sum(onperiod(:,1));
ns0 = 2^nr0; % number of associated states
SBsub = SB(1:ns0,1:nr0);
n_up_active = sum(SBsub(:,onperiod(inreg_all{1},1)),2); % number of initially active regions in up position for each state
n_down_active =  sum(~SBsub(:,onperiod(inreg_all{1},1)),2);  % number of initially active regions in down position for each state
xx0 = x0.^n_up_active .* (1-x0).^n_down_active; % initial probability for each state
this_alpha =  xx0.*pobs(1:ns0,1); % initialize alpha (bishop 13.37)
sum_alpha = sum(this_alpha);
alpha_normlog(1) = -log(sum_alpha) + pobs_normlog(1);

alpha(1:ns0,1) = this_alpha / sum_alpha;
for t=2:ntime
    oldA = expandbasis(alpha(:,t-1), onperiod(:,t-1:t), SB);
    this_AA = bsxfun(@times, oldA, AA(:,:,t-1)); % pass through transition matrix
    this_BB = zeros(ns(t-1),1);
    JJ = [1 inreg_all{t-1}'+1]; % first: no change; rest: change
    for i=1:ns(t-1)
        for j=1:nr(t-1)+1
            this_BB(i) = this_BB(i) + this_AA(TM(j,i),JJ(j)); %sum over all elements in transition matrix leading to that state
        end
    end
    this_alpha = this_BB .* pobs(1:ns(t-1),t); % bishop 13.36
    this_alpha = reducebasis(this_alpha, onperiod(:,t-1:t), SB);
    sum_alpha = sum(this_alpha);
    alpha_normlog(t) = alpha_normlog(t-1) - log(sum_alpha) +  pobs_normlog(t);
    alpha(1:2^nr(t-1),t) = this_alpha / sum_alpha;
end

beta(1:2^nr(ntime),ntime) = 1; % bishop 13.39
beta_normlog(ntime) = 0;
for t=ntime-1:-1:1
    newbeta = expandbasis(beta(:,t+1), onperiod(:,[t+1 t]), SB);
    this_BB = (newbeta .* pobs(:,t+1)); % bishop 13.38
    this_beta = zeros(ns(t),1);
    JJJ = [true false(1,nreg)]; % map new regions into entire set (including 1:no change)
    JJJ(inreg_all{t}+1) = true;
    for i=1:ns(t)
        this_beta(i) = sum(AA(i,JJJ,t) .* this_BB(TM(1:nr(t)+1,i))'); %sum over all elements in transition matrix coming from that state
    end
    this_beta = reducebasis(this_beta, onperiod(:,[t+1 t]), SB);
    sum_beta = sum(this_beta);
    beta_normlog(t) = beta_normlog(t+1) - log(sum_beta) +  pobs_normlog(t+1);
    beta(1:2^nr(t),t) = this_beta / sum_beta;
end

llh = - alpha_normlog(end); % from definition of alpha (bishop 13.34), LLH = log p(y1..yN) = log (sum_x alphaN(x))

% marginal and joint probabilities
gamma = zeros(nstate_max,ntime);
eta = zeros(nstate_max,nTS,ntime-1);

for t=1:ntime
    this_gamma = alpha(1:ns(t),t) .*  beta(1:ns(t),t); % bishop 13.33 (without normalization by p(Y) unncessary for maximization)
    gamma(1:ns(t),t) =  this_gamma/sum(this_gamma);
    if t<=ntime-1
        newbeta = expandbasis(beta(:,t+1), onperiod(:,[t+1 t]), SB);
        this_BB = (newbeta .* pobs(:,t+1)); % bishop 13.38
        this_eta = zeros(ns(t), nTS);
        oldA = expandbasis(alpha(:,t), onperiod(:,t:t+1), SB);
        JJJ = [true false(1,nreg)]; % map new regions into entire set (including 1:no change)
        JJJ(inreg_all{t}+1) = true;
        for i=1:ns(t)
            this_eta(i,JJJ) = oldA(i) * AA(i,JJJ,t) .* this_BB(TM(1:nr(t)+1,i))'; % bishop 13.43 (without normalization)
        end
        eta(1:ns(t),:,t) = this_eta /sum(this_eta(:));
    end
end

end % end of E-step function

%% compute transition matrix from one state to the other
function     [XX, grad, H] = p_trans(B, lambda, xi, E, G, dt, SB,eta,onperiod,x0, origin)
with_eta = ~isempty(eta); % if eta is provided (in M step), compute free energy and its gradient; otherwise (Estep) somply compute transition matrix
[R,nT,~] = size(E);
[lbd, with_ext] = lambdapar(lambda); % value of lambda, whether there is environemental variable

nreg_max = max(sum(onperiod(:,1:end-1) | onperiod(:,2:end))); % maximum number of regions active at a time
nstate_max = 2^nreg_max;

% compute transition matrix for each time step
n_lambdapar = length(lambda);
npar = 3+n_lambdapar; % number of parameters
if ~with_eta
    XX = zeros(nstate_max,R+1,nT-1); % pre x post for each time step
else
    XX = 0; % negative free-entropy
    if nargout>1
        grad = zeros(1,npar);
    end
    if nargout>2
        H = zeros(npar); % hessian
    end
end

rate = B; % if no environmental variable

for t=1:nT-1
    active = onperiod(:,t) & onperiod(:,t+1); % regions active for that time step and subsequent one
    starting = ~onperiod(:,t) & onperiod(:,t+1); %newly active regions
    stopping = onperiod(:,t) & ~onperiod(:,t+1); % last active time point
    
    inreg = find(active|starting|stopping); % index of active regions
    nr = length(inreg); % number of active states for this time step or next one
    ns = 2^nr;
    active = active(inreg);
    starting = starting(inreg);
    stopping = stopping(inreg);
    
    M = zeros(ns,R+1); % matrix of infinitesimal transition
    SBsub = SB(1:ns,1:nr);
    
    if nargout>1
        grad_M = zeros(ns,R+1,npar);
    end
    
    for r= find(active)'
        r2 = inreg(r); % corresponding index of entire set of regions
        if with_ext
            rate = B + lbd*E(r2,t); % rate of transition (from 0 to 1 and 1 to 0, resp.)
        end
        
        inactive_down = ~any(SBsub(:,~active),2);
        upstate = find(SBsub(:,r) & inactive_down); % find all states with region r up (and inactive regions in state down)
        downstate = find(~SBsub(:,r) & inactive_down); % find all states with region r down  (and inactive regions in state down)
        for i=1:length(upstate) %nstate/2
            % transition rate from down to up
            neighb = (2*SB(downstate(i),active)-1) * G(inreg(active),r); % upstate neighbours
            rr = rate(1) + xi*neighb;
            pos = rr>=0;
            rr = rr*pos; % if negative; turn rate to 0
            M(downstate(i),1+r2) = rr; % from down to up
            M(downstate(i),1) = M(downstate(i),1) - rr; % no-transition
            
            if nargout>1 % corresponding gradient
                grad_rr = zeros(1,1,npar);
                grad_rr(1:2) = [1 0]; % derivative over down-up and up-down fixed rate
                switch  n_lambdapar
                    case 1 % 3 paremters: fixedratetoup, fixedratetodown, symdeprate
                        grad_rr(3) = E(r,t);
                    case 2 % 4 parameters: fixedratetoup, fixedratetodown, depratetoup, depratetodown
                        grad_rr(3:4) = [E(r,t) 0];
                end
                grad_rr(end) = neighb;
                grad_rr = grad_rr*pos; % if negative rate, gradient is null
                
                grad_M(downstate(i),1+r2,: ) = grad_rr;
                grad_M(downstate(i),1,: ) = grad_M(downstate(i),1,: ) - grad_rr;
            end
            
            % transition rate from up to down
            rr = rate(2) - xi*neighb; % transition from up
            pos = rr>=0;
            rr = rr*pos; % if negative; turn rate to 0
            M(upstate(i),r2+1) = rr; % from up to down
            M(upstate(i),1) = M(upstate(i),1) -rr; % no transition
            
            if nargout>1 % corresponding gradient
                grad_rr = zeros(1,1,npar);
                grad_rr(1:2) = [0 1]; % derivative over down-up and up-down fixed rate
                switch  n_lambdapar
                    case 1 % 3 paremters: fixedratetoup, fixedratetodown, symdeprate
                        grad_rr(3) = - E(r2,t);
                    case 2 % 4 parameters: fixedratetoup, fixedratetodown, depratetoup, depratetodown
                        grad_rr(3:4) = [0 E(r2,t)];
                end
                grad_rr(end) = -neighb;
                grad_rr = grad_rr*pos; % if negative rate, gradient is null
                
                grad_M(upstate(i),r2+1,: ) = grad_rr;
                grad_M(upstate(i),1,: ) = grad_M(upstate(i),1,: ) - grad_rr;
            end
        end
        
    end
    
     AA = repmat([1 zeros(1,R)],ns,1) + dt(t)*M; % first order (if nstate
    if nargout>1
        grad_AA = dt(t)*grad_M;
    end
    
    % initial point for a region:
    for r= find(starting)'
        r2 = inreg(r);
        downstate = ~SBsub(:,r); % find all states with region r down
        if origin(r2) % if coming from scission of region
            org_rg = any(inreg==origin(r2)');
            if ~any(org_rg)
                error('region of origin not active');
            end
            upstate_org = SBsub(:,org_rg); % find all states with region of origin of r up
            AA(downstate & upstate_org,:) = 0; % if Origin in upstate and region in down state, all transitions to  0...
            AA(downstate & upstate_org,r2+1) = 1; % ... except down- to up-state in region r
            if nargout>1
                grad_AA(downstate & upstate_org,:,:) = 0;
            end
        else %starting from scratch: part proba into x0  for up-state, 1-x0
            % for down-state (we only split the no-transition states)
            AA(downstate,r+1) =  AA(downstate,1)*x0 + AA(downstate,r+1);
            AA(downstate,1) =  AA(downstate,1)*(1-x0);
            if nargout>1
                grad_AA(downstate,r+1,:) = grad_AA(downstate,1,:)*x0 + grad_AA(downstate,r+1,:);
                grad_AA(downstate,1,:) = grad_AA(downstate,1,:)*(1-x0);
            end
        end
    end
    
    %final point for a region: merge up-state and down-state into
    %down-state
    for r= find(stopping)'
        upstate = SBsub(:,r); % find all states with region r up
        AA(upstate,:) = 0; % we ignore all other possible transitions
        r2 = inreg(r);
        AA(upstate,r2+1) =   1;
        if nargout>1
            grad_AA(upstate,:,:) = 0;
        end
    end
    
    if ~with_eta
        XX(1:ns,:,t) = AA;
    elseif t<nT
        if any(AA(:)<0)
            XX = nan;
            grad = nan(size(grad));
            return;
        end
        QQ = eta(1:ns,:,t) .* log(AA);
        XX = XX - nansum(QQ(:));
        if isinf(nansum(QQ(:)))
            if any(any(AA==0 & eta(1:ns,:,t)~=0))
                %    warning('zero transmat at non-zero eta');
            else
                warning('weird inf');
            end
            if nargout>1
                grad = nan(size(grad));
                if nargout>2
                    H = nan(size(H));
                end
            end
            return;
        end
        if nargout>1
            grad_QQ = bsxfun(@times, eta(1:ns,:,t)./AA, grad_AA);
            grad = grad - permute(nansum(nansum(grad_QQ,1),2),[2 3 1]);
            if nargout>2
                H_QQ = - bsxfun(@times, bsxfun(@times, grad_QQ, 1./AA), permute(grad_AA, [1 2 4 3]));
                H = H - squeeze(nansum(nansum(H_QQ,1),2));
            end
        end
    end
    
end
end

%% negative of Q function that is maximized in M step (only dependence on transition vars)
function [Qneg, grad, H] = negQtransition(x, eta, E, G, dt,SB,onperiod,x0, origin)
B = x(1:2);
lambda = x(3:end-1);
xi = x(end);

% compute transition matrix at each time step and its gradient
[Qneg, grad, H] = p_trans(B, lambda, xi, E, G, dt, SB,eta,onperiod,x0, origin);
end




%% compute marginal probabilities for up or down-state for one region
function Q = gamma_marg(gamma, SB, onperiod)
R = size(SB,2);
nT = size(gamma,2);
Q = zeros(R,nT);

for t=1:nT
    if t>1
        inreg = find(onperiod(:,t-1) | onperiod(:,t)); % regions active for that time step and subsequent one
    else
        inreg = find(onperiod(:,t)); % regions active for that time step and subsequent one
    end
    
    nr = length(inreg); % number of region for this time step
    ns = 2^nr; % number of associated states
    SBsub = SB(1:ns,1:nr); % sub matrix from states to region active/passive
    
    for r=1:nr % for each region
        Q(inreg(r),t) = sum(gamma(SBsub(:,r),t),1); % sum over all states where this regions is active
    end
end
end

%% get from lambda parameter to values of state-dependent transition for both types of transition
function  [lbd, with_ext, lambda_sym] = lambdapar(lambda)
with_ext = ~isempty(lambda);
lambda_sym =(length(lambda)==1);
if with_ext % just check no negative value
    if lambda_sym
        lbd = [lambda -lambda];
    else
        lbd = lambda;
    end
else
    lbd = [];
end
end

%% get initial and last active time points for each region
function [starttime, stoptime] = startstoptime(onperiod)
nreg = size(onperiod,1);
starttime = zeros(1,nreg);
stoptime = zeros(1,nreg);
for r=1:nreg
    starttime(r) = find( onperiod(r,:),1); % initial point for each region
    stoptime(r) = find( onperiod(r,:),1,'last'); % initial point for each region
end
end

%%
function l = xlogy(A,B,amin)
S = A .* log(B);
S(A<=amin) = 0; % remove zeros values to avoid problems with 0 * log(0)
S(isnan(A)) = 0;
l = sum(S(:));
end


%% adapt basis of transition matrix to integrate new states
function A = expandbasis(A, inreg, SB)
old = inreg(:,1);
new = inreg(:,2) | old;
in_old = any(old'== new,2); % whether each new is in the old one
if all(in_old) % if all were already there, dont change
    return;
end
old_id = old(new); % id of old regions in new set
new_id = ~old_id;
n_ns = 2^sum(new); % number of states in new basis

SB = SB(1:n_ns,:);  % extract state - region matrix

AA = zeros(n_ns, size(A,2)); % create new with all zeros

idx = all(~SB(:,new_id)); % index of states already represented correspond to 0 for new regions
AA(idx,:) = A; %
A = AA;
end

% adapt basis of transition matrix to remove old states
function A = reducebasis(A, inreg, SB)
new = inreg(:,2);
old = inreg(:,1) | new;
in_new = any(new'== old,2); % whether each old is in the new one
if all(in_new) % if all are still there, dont change
    return;
end
old_id = new(old); % id of old regions not included in new set
n_os = 2^sum(new); % number of states in old basis

SB = SB(1:n_os,:);  % extract state - region matrix

idx = all(~SB(:,old_id)); % index of states to be discarded correspond to 0 for to be abandonned regions
if any(any(A(~idx,:)==0))
    error('sth wrong here');
end
A = A(idx,:); % extract
end

%% EM algorithm
function [C,x0,B,lambda,xi,exitflag_Mtrans, llh] = EMalgo(R, nT,E, dt, A, G, B, lambda, x0, xi, C, SB,onperiod, origin, with_ext, lambda_sym, dt_notlast, TolFun, maxiter, Y_on,idx,npar, verbose)

iter=1; % number of EM iterations
    tic;
    while iter<3 || ((~(Fdiff < TolFun) || (exitflag_Mtrans<=0) ) && (iter<=maxiter)) % loop until convergence of maximum number of iterations is reached
        if lambda_sym
            lbd = [lambda -lambda];
        else
            lbd = lambda;
        end
        if with_ext && any(any(bsxfun(@plus,E(:)*lbd,B)<0))
            warning('negative rate');
        end
        
        %% Estep: forward-backward algorithm
        [llh, gamma, eta] = Estep(R, nT, E, dt, A, G, B, lambda, x0, xi, C,SB,onperiod, origin);
        
        if any(isnan(eta(:)))
            error('nan eta');
        end
        if iter > 1 && (llh<Fm)
            warning('Free energy decreased after E step by %f', Fm-llh); % how much free energy increased from this E-step
        end
        
        %% M-step
        [C,x0,B,lambda,xi,new_F,old_F,exitflag_Mtrans] = Mstep(C,x0,B,lambda,xi,  gamma, eta, R, E, G, dt_notlast,SB,Y_on,TolFun,idx,npar,onperiod, origin);
        
        Fm = llh + sum(new_F)- sum(old_F); % sum all change of contributions to Q (adapted from Bishop 9.71 when only changing parameters)
        
        if llh>Fm
            warning('Free energy decreased after M step by %f', llh-Fm); % how much free energy increased from this E-step
        end
        if iter > 1
            Fdiff = Fm - Fm_prev; % how much free energy increased from latest iteration
            if Fdiff<0
                warning('Fm going wrong direction');
                Fdiff = 2*TolFun;
            end
        end
        Fm_prev = Fm;
        
        if verbose
            n_lambdapar = length(lambda);
            fprintf(['iter %d, Fe %f, Fm %f, A %f, %f, lambda '  repmat(' %f',1,n_lambdapar) ', xi %f, C %f, %f, x0 %f\n'],...
                iter,llh, Fm, B(1), B(2), lambda, xi, C(1),C(2), x0);
        end
        
        iter = iter+1;
    end
    toc;
end
