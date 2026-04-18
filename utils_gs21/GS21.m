%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Firm price adjustment problem
% Setting up grids and discretization and payoff matrices
% Firm_Pricing_Setup.m
%
% S. Terry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%clear; close all; clc;

disp('%%%%%%%%%%%%% Solving the firm price adjustment problem')
disp(' ')

%% set model parameters
beta = 0.994;
varphi = 2;
gamma = 10;

tol = 1e-4;
g = 1.14 ; %(1.14-1)/3 + 1;
alpha = 0.2;
delta = 0.02/3;
rho_x = 0.95^(1/3);
sigma_x = 0.012*sqrt((1 - 0.95^(2/3))/(1 - 0.95^2));
rho_z = 0.9^(1/3);
sigma_z = 0.16*sqrt((1 - 0.9^(2/3))/(1 - 0.9^2));
chi = 1;
r = 0.1/12; %0.074830/12; %0.2;
gamma_x = 0.5;
x_bar = 0; 

tau = 0.2/3;
phi = 0.4;
kappa_e = 0; %0.025;
kappa_b = 0.004;
xi = 0.03/3*3;

%% set solution parameters

tic
% grid parameters
bmin = 0.00; %lowest grid point for debt
bmax = 1; %highest grid point for debt
bnum = 20; %number of debt grid points

imin = 0.0; %lowest grid point for investment cost
imax = 2000; %highest grid point for investment cost
inum = 40;

xnum = 20; %number of x grid points
znum = 200; %number of z grid points
mnstdev = 4; %number of standard deviations to cover 

sigma_m = 5; % 2.5
mbar = 4*sigma_m;
[nodes, w] = gaussHermite(20000);
w= w/sqrt(pi);
nodes = nodes*sqrt(2)*sigma_m;
weights = w((nodes > -mbar) &(nodes < mbar));
nodes = nodes((nodes > -mbar) &(nodes < mbar));
weights = weights/sum(weights);

%%%%%%%%%% set up (log)linear grid for endogenous debt and investment

%log_bgrid = log(bmin):log(g):log(bmax);
bgrid = linspace(bmin, bmax, bnum)';

igrid = linspace(imin, imax, inum+1)'; 
igrid = igrid(1:end-1);

%%%%%%%%%% set up discretization for x and z AR(1) processes


[xgrid, pr_mat_x] = tauchen(sigma_x, rho_x, mnstdev, xnum);
[zgrid, pr_mat_z] = tauchen(sigma_z, rho_z, mnstdev, znum);
pr_mat = kron(pr_mat_z, pr_mat_x); % (xnum*znum) x (xnum*znum) transition matrix
pr_mat_re = kron(pr_mat, ones(bnum,1)); % dimensions adjusted for debt state

statenum = bnum*xnum*znum; %total number of state grid points

%solution parameters
tol_debt = 1e-3; %tolerance on model debt solution
tol_price = 1e-4; %tolerance on model price solution
maxit = 1000; %max iterations on model solution





%% set up joint indexes of the state space

grid_val = zeros(statenum,3); %bnum*xnum*znum x 3, with (i,1) = value of b, (i,2) = value of x, (i,3) = value of z
grid_ind = zeros(statenum,3); %bnum*xnum*znum x 3, with (i,1) = index of b, (i,2) = index of x, (i,3) = index of z

%insert values
grid_val(:,1) = kron(ones(xnum*znum,1),bgrid);
grid_val(:,2) = kron(ones(znum, 1), kron(xgrid,ones(bnum,1)));
grid_val(:,3) = kron(zgrid,ones(bnum*xnum,1));

%insert indexes
grid_ind(:,1) = kron(ones(xnum*znum,1),(1:bnum)');
grid_ind(:,2) = kron(ones(znum, 1), kron((1:xnum)',ones(bnum,1)));
grid_ind(:,3) = kron((1:znum)', ones(bnum*xnum,1));

% %%%%% Updated grid for when i is also a state variable
% grid_val_i = zeros(statenum*inum,4); %inum*bnum*xnum*znum x 4, with (i,1) = value of i, (i,2) = value of b, (i,3) = value of x, (i,4) = value of z
% grid_ind_i = zeros(statenum*inum,4); %inum*bnum*xnum*znum x 4, with (i,1) = index of i, (i,2) = index of b, (i,3) = index of x, (i,4) = value of z
% 
% %insert values
% grid_val_i(:,1) = kron(ones(bnum*xnum*znum,1),igrid);
% grid_val_i(:,2) = kron(ones(xnum*znum, 1), kron(bgrid, ones(inum,1)));
% grid_val_i(:,3) = kron(ones(znum, 1), kron(xgrid, ones(bnum*inum,1)));
% grid_val_i(:,4) = kron(zgrid,ones(bnum*xnum*inum,1));
% 
% %insert indexes
% grid_ind_i(:,1) = kron(ones(bnum*xnum*znum,1),(1:inum)');
% grid_ind_i(:,2) = kron(ones(xnum*znum, 1), kron((1:bnum)', ones(inum,1)));
% grid_ind_i(:,3) = kron(ones(znum, 1), kron((1:xnum)', ones(bnum*inum,1)));
% grid_ind_i(:,4) = kron((1:znum)',ones(bnum*xnum*inum,1));
% 
% i_grid_re = reshape(grid_val_i(:, 1), inum, bnum*xnum*znum)'; % reshaped grid for computing expected prices over i
z_grid_re = reshape(grid_val(:,3), bnum, xnum*znum);  % z grid reshape
    
%% set up payoff matrices

% payoffs not including debt 
% statnum x 1 vectors of payoff for each state variable on grid
pi_Rmat   = (exp(grid_val(:,2)+grid_val(:,3)) - delta) - (1-tau).*grid_val(:,1);
%pi_Rmat_i = (exp(grid_val_i(:,3)+grid_val_i(:,4)) - delta) ...
%            - grid_val_i(:,1) - (1-tau).*grid_val_i(:,2);

% liquidation value after default
% stanum x (xnum*znum) payoff. Interacted with default decision below
def_Rmat_pre = phi*(1 - delta + exp(reshape(grid_val(:,2)+grid_val(:,3), bnum, xnum*znum))) ;
def_Rmat = repmat(phi*(1 - delta + exp(reshape(grid_val(:,2)+grid_val(:,3), bnum, xnum*znum))) , xnum*znum, 1);

% sdf matrix
% (statnum) x (xnum*znum)
Mmat=  exp(-r - 0.5*gamma_x^2 - gamma_x*(xgrid' - (1- rho_x)*x_bar - rho_x*grid_val(:, 2))/sigma_x);
Mmat = repmat(Mmat, 1, znum);

toc
disp('%%% Set up grids and discretization and payoffs')
disp(' ')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Gomes Schmid 21 problem
% Implementing baseline VFI solution
% Firm_Pricing_VFI.m
%
% code modified based on code from S. Terry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('%%% Implement baseline VFI solution')
disp(' ')
tic

%% do the VFI loop

%initialize guess for value function
P_up_old = zeros(statenum,1);
P_down_old = zeros(statenum,1);
Q_up_old = zeros(statenum,1);
Q_0_old = zeros(statenum,1);
Q_down_old = zeros(statenum,1);
z_cut_up = zgrid(ones(xnum*bnum, 1));
weight_cut_up = 0.5*(ones(xnum*bnum, 1));
z_cut_down = zgrid(ones(xnum*bnum, 1));
weight_cut_down = 0.5*(ones(xnum*bnum, 1));
Q_I_re_old = zeros(statenum, 1);
grid_up = 0; grid_down = 0;
prob_i_up = 1; prob_i_down = 1;
for vfct = 1:maxit
    %% --- Prepare reshaped variables once ---
    zcut_up_re   = reshape(z_cut_up,   bnum, xnum);
    zcut_down_re = reshape(z_cut_down, bnum, xnum);
    
    % default cutoffs
    I_up   = (z_grid_re >= repmat(zcut_up_re,   1, znum));
    I_down = (z_grid_re >= repmat(zcut_down_re, 1, znum));

    I_up_re = repmat(I_up, xnum*znum, 1);
    I_down_re = repmat(I_down, xnum*znum, 1);
    
    w_up_re      = repmat(reshape(weight_cut_up -0.5,   bnum, xnum), 1, znum);
    w_down_re    = repmat(reshape(weight_cut_down -0.5, bnum, xnum), 1, znum);
    w_up = repmat(((z_grid_re == repmat(zcut_up_re,   1, znum)) & (z_grid_re > zgrid(1))).*w_up_re, xnum*znum, 1);
    w_down = repmat(((z_grid_re == repmat(zcut_down_re,   1, znum)) & (z_grid_re > zgrid(1))).*w_down_re, xnum*znum, 1);
    w_up_lower = repmat(-((z_grid_re == repmat(zcut_up_re-1,   1, znum)) & (z_grid_re > zgrid(1))).*w_up_re, xnum*znum, 1);
    w_down_lower = repmat(-((z_grid_re == repmat(zcut_down_re-1,   1, znum)) & (z_grid_re > zgrid(1))).*w_down_re, xnum*znum, 1);
    
    for vfct_debt = 1:maxit
        
        %% --- Update Q_up and Q_down ---
        
        Q_up_old   = prob_i_up.*Q_I_re_old(:)   + (1 - prob_i_up).*Q_0_old;
        Q_down_old = prob_i_down.*Q_I_re_old(:) + (1 - prob_i_down).*Q_0_old;
        
        
        %% --- Compute Q_0 expectations ---
        % Up region
        
        

        EQ0_up = sum(((grid_val(:, 1) + Q_up_old).* ((P_up_old + nodes') > 0) + def_Rmat_pre(:).* ((P_up_old + nodes') <= 0)).*(weights.'), 2);
        %Q0_solv_up = (grid_val(:,1) + repmat(reshape(Q_up_old, bnum, xnum*znum), xnum*znum, 1)).*I_up_re.*(1 + 0*w_up); 
        %Qdef_up = def_Rmat.*(1 - I_up_re).*(1 + 0*w_up_lower);
        EQ0_up = sum(Mmat .* (repmat(reshape(EQ0_up, bnum, xnum*znum), xnum*znum, 1)) .* pr_mat_re, 2);
        
        % Down region
        EQ0_down = sum(((grid_val(:, 1) + Q_down_old).* ((P_down_old + nodes') > 0) + def_Rmat_pre(:).* ((P_down_old + nodes') <= 0)).*(weights.'), 2);
        %Q0_solv_down = (grid_val(:,1) + repmat(reshape(Q_down_old, bnum, xnum*znum), xnum*znum, 1)).* I_down_re.*(1 + 0*w_down);
        %Q0_def_down = def_Rmat.* (1 - I_down_re).*(1 + 0*w_down_lower);
        EQ0_down = sum(Mmat .* repmat(reshape(EQ0_down, bnum, xnum*znum), xnum*znum, 1) .* pr_mat_re, 2);
        
        % Combine
        Q_0 = xi*EQ0_up + (1-xi)*EQ0_down;
        
        %% --- Interpolate Q_I ---
        
        Q_I = g*Q_0;
        Q_I_re = reshape(Q_I, bnum, xnum*znum);
        Q_I_no = interp1(bgrid, Q_I_re, bgrid/g, 'spline', 'extrap'); % Q_I assuming tilde b stays the same
        

        %% --- Update old values ---
        solerr = max(abs((Q_0(:)-Q_0_old(:))));
        disp(['Iter ' num2str(vfct) ', Q Iter ' num2str(vfct_debt) ',  VF Q error = ' num2str(solerr)])
        
        if mod(vfct_debt, 1) == 0
            res = reshape(Q_0,bnum,znum*xnum);
            figure(1); hold on; plot(bgrid, res(:,znum*xnum/2 + xnum/2));
        end
        if solerr < tol_debt
            tol_debt = tol_debt/1.2;
            break
            
        end
        
        Q_0_old   = Q_0;
        Q_I_re_old = Q_I_no;
                                                                            
    end
    
    for vfct_price = 1:maxit

        %w_up_re      = 2*repmat(reshape(weight_cut_up -0.5,   bnum, xnum), 1, znum);
        %w_down_re    = 2*repmat(reshape(weight_cut_down -0.5, bnum, xnum), 1, znum);
        %w_up = repmat(((z_grid_re == repmat(zcut_up_re,   1, znum)) & (z_grid_re > zgrid(1))).*(0.25*(2 + w_up_re).^2./(1 + w_up_re) - 1), xnum*znum, 1);
        %w_down = repmat(((z_grid_re == repmat(zcut_down_re,   1, znum)) & (z_grid_re > zgrid(1))).*(0.25*(2 + w_down_re).^2./(1 + w_down_re) - 1), xnum*znum, 1);

        %% --- Profits and prices ---
        prof0_up  = pi_Rmat + ((1-kappa_b)*kron(reshape(Q_0,bnum,xnum*znum)',ones(bnum,1)) - Q_0);
        P0_up_R   = (1 + (prof0_up<=0).*kappa_e).*prof0_up;
        P0_down_R = (1 + (pi_Rmat<=0).*kappa_e).*pi_Rmat;
        
        
        profI_up  = pi_Rmat + ((1-kappa_b)*kron(Q_I_re',ones(bnum,1)) ...
            - Q_I_no(:));
        PI_up_R   = (1 + (profI_up<=0).*kappa_e).*profI_up;
        PI_down_R = (1 + (pi_Rmat<=0).*kappa_e).*pi_Rmat;
        
        %% --- Expectation terms EPI and EP0 ---
        P_up_old_pert = sum(max(P_up_old + nodes', 0).*(weights.'), 2);
        P_down_old_pert = sum(max(P_down_old + nodes', 0).*(weights.'), 2);
        P_up_old_re   = reshape(P_up_old_pert,   bnum, xnum*znum);
        P_down_old_re = reshape(P_down_old_pert, bnum, xnum*znum);
        
        % EPI
        EPI_up   = g*P_up_old_re;
        EPI_up   = sum(Mmat.*repmat(EPI_up,  znum*xnum,1).*pr_mat_re,2);
        EPI_down = g*P_down_old_re;
        EPI_down = sum(Mmat.*repmat(EPI_down,znum*xnum,1).*pr_mat_re,2);
        EPI = xi*EPI_up + (1-xi)*EPI_down;
        
        % EP0
        EP0_up   = P_up_old_re ;
        EP0_up   = sum(Mmat.*repmat(EP0_up,  znum*xnum,1).*pr_mat_re,2);
        EP0_down = P_down_old_re ;
        EP0_down = sum(Mmat.*repmat(EP0_down,znum*xnum,1).*pr_mat_re,2);
        EP0 = xi*EP0_up + (1-xi)*EP0_down;
        
        EP0 = kron(reshape(EP0,bnum,xnum*znum)',ones(bnum,1));
        EPI = kron(reshape(EPI,bnum,xnum*znum)',ones(bnum,1));
        
        %% --- Construct P0 and PI matrices ---
        P0_up   = P0_up_R   + EP0;
        P0_down = P0_down_R + EP0;
        PI_up   = PI_up_R   + EPI;
        PI_down = PI_down_R + EPI;
        
        N = size(P0_up,1);
        
        [P0_up,   no_up_bprime]   = max(P0_up,[],2);
        P0_down = interp1(bgrid, P0_down', bgrid, 'spline', 'extrap')';
        P0_down = P0_down(sub2ind(size(P0_down), (1:size(P0_up,1)).', grid_ind(:, 1)));
        [PI_up,   I_up_bprime]    = max(PI_up,[],2);
        PI_down = interp1(bgrid, PI_down', bgrid/g, 'spline', 'extrap')';
        PI_down = PI_down(sub2ind(size(PI_down), (1:size(PI_down,1)).', grid_ind(:, 1)));
        
        %% --- Solve for i_cut_up/down ---
        %cond_up   = reshape(PI_up,   inum,bnum*xnum*znum)' <= P0_up;
        %[~,ind_up]= max(cond_up,[],2);
        %i_cut_up  = (sum(cond_up,2)==0)*imax ...
        %    + (sum(1-cond_up,2)==0)*imin ...
        %    + (1-(sum(cond_up,2)==0)-(sum(1-cond_up,2)==0)).*igrid(ind_up);
        
        %cond_down   = reshape(PI_down,inum,bnum*xnum*znum)' <= P0_down;
        %[~,ind_down]= max(cond_down,[],2);
        %i_cut_down  = (sum(cond_down,2)==0)*imax ...
        %    + (sum(1-cond_down,2)==0)*imin ...
        %    + (1-(sum(cond_down,2)==0)-(sum(1-cond_down,2)==0)).*igrid(ind_down);
        
        i_cut_up =  min(imax, max(imin, (PI_up - P0_up)));
        i_cut_down =  min(imax, max(imin, (PI_down - P0_down)));

        %% --- Update P_up ---
        %grid_up = (i_grid_re >= i_cut_up);
        %P_up_re = reshape(PI_up,inum,bnum*xnum*znum)';
        %P_up = sum(grid_up.*P0_up + (1-grid_up).*reshape(PI_up,inum,bnum*xnum*znum)',2)/inum;
        
        prob_i_up = (i_cut_up - imin)/(imax - imin);
        P_up = prob_i_up.*(PI_up - 0.5*(i_cut_up + imin)) + (1 - prob_i_up).*P0_up;

        %i_cut_up = max(0, min(imax, (mean(P_up_re, 2) - P0_down)));
        %P_up = ((imax - i_cut_up).*P0_up + i_cut_up.*(mean(P_up_re,2) - i_cut_up/imax))/imax;
        
        [z_cut_up, weight_cut_up] = update_cutoffs(P_up,zgrid,bnum,xnum,znum);
        
        
        %% --- Update P_down ---
        %grid_down = (i_grid_re >= i_cut_down);
        %PI_down_re = reshape(PI_down,inum,bnum*xnum*znum)';
        %P_down = sum(grid_down.*P0_down + (1-grid_down).*PI_down_re,2)/inum;
        
        prob_i_down = (i_cut_down - imin)/(imax - imin);
        P_down = prob_i_down.*(PI_down - 0.5*(i_cut_down + imin)) + (1 - prob_i_down).*P0_down;

        %i_cut_down = max(0, min(imax, (mean(PI_down_re, 2) - P0_down)));
        %P_up = ((imax - i_cut_up).*P0_down + i_cut_up.*(mean(PI_down_re,2) - i_cut_down/imax))/imax;
        
        [z_cut_down, weight_cut_down] = update_cutoffs(P_down,zgrid,bnum,xnum,znum);
        
        %z_cut_up = 0*z_cut_up -1;
        %z_cut_down = 0*z_cut_down -1;
        
        %% --- Error checking and diagnostics ---
        solerr = max(abs((P_down(:)-P_down_old(:))));
        if mod(vfct_price,1)==0
            disp(['Iter ' num2str(vfct) ', P Iter ' num2str(vfct_price) ', VF P error = ' num2str(solerr)])
            if mod(vfct_price, 5) == 1
                tmp = reshape(P_down,bnum,xnum*znum);
                res = reshape(tmp(5,:),xnum,znum);
                figure(2); hold on; plot(xgrid, res(:,znum/2));
                figure(3); hold on; plot(zgrid, res(xnum/2, :));
            end
        end
        
        if solerr<tol_price 
            tol_price = tol_price/1.2;
            break
        end
        P_up_old = P_up;
        P_down_old = P_down;
        
    end
    
    
    
    
    
end
writematrix(zgrid, 'zgrid.csv');
writematrix(xgrid, 'xgrid.csv');
writematrix(igrid, 'igrid.csv');
writematrix(bgrid, 'bgrid.csv');
writematrix(z_cut_up, 'z_cut_up.csv');
writematrix(z_cut_down, 'z_cut_down.csv');
writematrix(i_cut_up, 'i_cut_up.csv');
writematrix(i_cut_down, 'i_cut_down.csv');
writematrix(Q_I, 'Q_I.csv');
writematrix(Q_0, 'Q_0.csv');
writematrix(P_up, 'P_up.csv');
writematrix(P_down, 'P_down.csv');
writematrix(PI_up, 'PI_up.csv');
writematrix(PI_down, 'PI_down.csv');
writematrix(P0_up, 'P0_up.csv');
writematrix(P0_down, 'P0_down.csv');
writematrix(bgrid(no_up_bprime), 'b_refin_0.csv');
writematrix(bgrid(I_up_bprime), 'b_refin_I.csv');




%% --- Helper for cutoffs ---
function [z_cut, weight_cut] = update_cutoffs(P,zgrid,bnum,xnum,znum)
    pos_val = reshape(P,xnum*bnum,znum) > 0;
    [~, z_ind] = max(pos_val,[],2);
    cond0 = (sum(pos_val,2)==0);
    z_cut = cond0.*(zgrid(end)+1e-10) + (1-cond0).*zgrid(z_ind);
    
    P_plus  = P(sub2ind(size(reshape(P,bnum*xnum,znum)), (1:bnum*xnum)', z_ind));
    P_minus = P(sub2ind(size(reshape(P,bnum*xnum,znum)), (1:bnum*xnum)', max(1,z_ind-1)));
    cond2   = (sum(1-pos_val,2)==0);
    weight = P_plus./(P_plus-P_minus);
    weight(isnan(weight) | isinf(weight)) = 0.5;
    weight_cut = cond0*0.5 + cond2*0.5 + (1-cond0-cond2).*weight;
    %P = max(0,P);
end

function [x, w] = gaussHermite(n)
    % Compute Gauss-Hermite nodes and weights for n-point quadrature.
    % Integral approximation: ∫ exp(-x^2) f(x) dx ≈ sum(w .* f(x))

    % Beta coefficients for Hermite polynomials
    i = (1:n-1)';
    beta = sqrt(i/2);

    % Build symmetric tridiagonal matrix
    J = diag(beta,1) + diag(beta,-1);

    % Eigen-decomposition
    [V, D] = eig(J);

    % Nodes are eigenvalues
    x = diag(D);

    % Weights (squared first component of eigenvectors * sqrt(pi))
    w = V(1,:).^2 * sqrt(pi);

    % Ensure column vectors
    x = x(:);
    w = w(:);
end