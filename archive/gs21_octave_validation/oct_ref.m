% One debt pass + one price pass using GS21.m's expressions VERBATIM (dense form).
% Reads the grids / transition matrices / GH nodes / random inputs that py_ref.py
% dumped, so any difference against the Python port is algebra, not setup.
warning('off','all');
here = fileparts(mfilename('fullpath'));
rd = @(n) dlmread(fullfile(here, [n '.csv']), ',');
run(fullfile(here, 'params.m'));

bgrid = rd('bgrid'); xgrid = rd('xgrid'); zgrid = rd('zgrid');
pr_mat_x = rd('pr_x'); pr_mat_z = rd('pr_z');
nodes = rd('nodes'); weights = rd('weights');
P_up_old = rd('in_P_up_old'); P_down_old = rd('in_P_down_old');
Q_0_old = rd('in_Q_0_old'); Q_I_re_old = rd('in_Q_I_re_old');
prob_i_up = rd('in_prob_i_up'); prob_i_down = rd('in_prob_i_down');

pr_mat = kron(pr_mat_z, pr_mat_x);
pr_mat_re = kron(pr_mat, ones(bnum,1));
statenum = bnum*xnum*znum;

% --- state space: GS21.m:95-102 verbatim ---
grid_val = zeros(statenum,3); grid_ind = zeros(statenum,3);
grid_val(:,1) = kron(ones(xnum*znum,1),bgrid);
grid_val(:,2) = kron(ones(znum, 1), kron(xgrid,ones(bnum,1)));
grid_val(:,3) = kron(zgrid,ones(bnum*xnum,1));
grid_ind(:,1) = kron(ones(xnum*znum,1),(1:bnum)');
grid_ind(:,2) = kron(ones(znum, 1), kron((1:xnum)',ones(bnum,1)));
grid_ind(:,3) = kron((1:znum)', ones(bnum*xnum,1));

% --- payoffs / SDF: GS21.m:127,133,138-139 verbatim ---
pi_Rmat = (exp(grid_val(:,2)+grid_val(:,3)) - delta) - (1-tau).*grid_val(:,1);
def_Rmat_pre = phi*(1 - delta + exp(reshape(grid_val(:,2)+grid_val(:,3), bnum, xnum*znum)));
Mmat = exp(-r - 0.5*gamma_x^2 - gamma_x*(xgrid' - (1- rho_x)*x_bar - rho_x*grid_val(:, 2))/sigma_x);
Mmat = repmat(Mmat, 1, znum);

% ======================= debt pass: GS21.m:195-222 =======================
Q_up_old   = prob_i_up.*Q_I_re_old(:)   + (1 - prob_i_up).*Q_0_old;
Q_down_old = prob_i_down.*Q_I_re_old(:) + (1 - prob_i_down).*Q_0_old;

EQ0_up = sum(((grid_val(:, 1) + Q_up_old).* ((P_up_old + nodes') > 0) + def_Rmat_pre(:).* ((P_up_old + nodes') <= 0)).*(weights.'), 2);
pay_up = EQ0_up;
EQ0_up = sum(Mmat .* (repmat(reshape(EQ0_up, bnum, xnum*znum), xnum*znum, 1)) .* pr_mat_re, 2);

EQ0_down = sum(((grid_val(:, 1) + Q_down_old).* ((P_down_old + nodes') > 0) + def_Rmat_pre(:).* ((P_down_old + nodes') <= 0)).*(weights.'), 2);
pay_dn = EQ0_down;
EQ0_down = sum(Mmat .* repmat(reshape(EQ0_down, bnum, xnum*znum), xnum*znum, 1) .* pr_mat_re, 2);

Q_0 = xi*EQ0_up + (1-xi)*EQ0_down;
Q_I = g*Q_0;
Q_I_re = reshape(Q_I, bnum, xnum*znum);
Q_I_no = interp1(bgrid, Q_I_re, bgrid/g, 'spline', 'extrap');

% ===================== price pass: GS21.m:252-341 =====================
prof0_up  = pi_Rmat + ((1-kappa_b)*kron(reshape(Q_0,bnum,xnum*znum)',ones(bnum,1)) - Q_0);
P0_up_R   = (1 + (prof0_up<=0).*kappa_e).*prof0_up;
P0_down_R = (1 + (pi_Rmat<=0).*kappa_e).*pi_Rmat;
% NOTE GS21.m:257 uses Q_I_re (= reshape(g*Q_0), the b'-CHOICE term) here while
% subtracting Q_I_no (the spline-shifted current-debt term). Two different objects.
profI_up  = pi_Rmat + ((1-kappa_b)*kron(Q_I_re',ones(bnum,1)) - Q_I_no(:));
PI_up_R   = (1 + (profI_up<=0).*kappa_e).*profI_up;
PI_down_R = (1 + (pi_Rmat<=0).*kappa_e).*pi_Rmat;

P_up_old_pert   = sum(max(P_up_old + nodes', 0).*(weights.'), 2);
P_down_old_pert = sum(max(P_down_old + nodes', 0).*(weights.'), 2);
P_up_old_re   = reshape(P_up_old_pert,   bnum, xnum*znum);
P_down_old_re = reshape(P_down_old_pert, bnum, xnum*znum);

EPI_up   = g*P_up_old_re;   EPI_up   = sum(Mmat.*repmat(EPI_up,  znum*xnum,1).*pr_mat_re,2);
EPI_down = g*P_down_old_re; EPI_down = sum(Mmat.*repmat(EPI_down,znum*xnum,1).*pr_mat_re,2);
EPI = xi*EPI_up + (1-xi)*EPI_down;
EP0_up   = P_up_old_re;     EP0_up   = sum(Mmat.*repmat(EP0_up,  znum*xnum,1).*pr_mat_re,2);
EP0_down = P_down_old_re;   EP0_down = sum(Mmat.*repmat(EP0_down,znum*xnum,1).*pr_mat_re,2);
EP0 = xi*EP0_up + (1-xi)*EP0_down;
EPI_v = EPI; EP0_v = EP0;
EP0 = kron(reshape(EP0,bnum,xnum*znum)',ones(bnum,1));
EPI = kron(reshape(EPI,bnum,xnum*znum)',ones(bnum,1));

P0_up   = P0_up_R   + EP0;  P0_down = P0_down_R + EP0;
PI_up   = PI_up_R   + EPI;  PI_down = PI_down_R + EPI;

[P0_up,   no_up_bprime]   = max(P0_up,[],2);
P0_down = interp1(bgrid, P0_down', bgrid, 'spline', 'extrap')';
P0_down = P0_down(sub2ind(size(P0_down), (1:size(P0_up,1)).', grid_ind(:, 1)));
[PI_up,   I_up_bprime]    = max(PI_up,[],2);
PI_down = interp1(bgrid, PI_down', bgrid/g, 'spline', 'extrap')';
PI_down = PI_down(sub2ind(size(PI_down), (1:size(PI_down,1)).', grid_ind(:, 1)));

i_cut_up   = min(imax, max(imin, (PI_up - P0_up)));
i_cut_down = min(imax, max(imin, (PI_down - P0_down)));
prob_i_up_n   = (i_cut_up - imin)/(imax - imin);
P_up   = prob_i_up_n.*(PI_up - 0.5*(i_cut_up + imin)) + (1 - prob_i_up_n).*P0_up;
prob_i_down_n = (i_cut_down - imin)/(imax - imin);
P_down = prob_i_down_n.*(PI_down - 0.5*(i_cut_down + imin)) + (1 - prob_i_down_n).*P0_down;

[z_cut_up, weight_cut_up]     = update_cutoffs(P_up,  zgrid,bnum,xnum,znum);
[z_cut_down, weight_cut_down] = update_cutoffs(P_down,zgrid,bnum,xnum,znum);

wr = @(n,a) dlmwrite(fullfile(here,['oct_' n '.csv']), a, 'delimiter',',','precision','%.17g');
wr('Q_up_old',Q_up_old); wr('Q_down_old',Q_down_old);
wr('pay_up',pay_up); wr('pay_dn',pay_dn);
wr('EQ0_up',EQ0_up); wr('EQ0_down',EQ0_down);
wr('Q_0',Q_0); wr('Q_I_no',Q_I_no(:));
wr('P_up_pert',P_up_old_pert); wr('P_down_pert',P_down_old_pert);
wr('EPI',EPI_v); wr('EP0',EP0_v);
wr('P0_up',P0_up); wr('P0_down',P0_down); wr('PI_up',PI_up); wr('PI_down',PI_down);
wr('i_cut_up',i_cut_up); wr('i_cut_down',i_cut_down);
wr('P_up',P_up); wr('P_down',P_down);
wr('b_refin_0',bgrid(no_up_bprime)); wr('b_refin_I',bgrid(I_up_bprime));
wr('z_cut_up',z_cut_up); wr('weight_cut_up',weight_cut_up);
wr('z_cut_down',z_cut_down); wr('weight_cut_down',weight_cut_down);
printf('octave single-pass done: statenum=%d\n', statenum);
