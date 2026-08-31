% GS21.m's FULL nested VFI loop, verbatim (disp/figure stripped, outer count
% parametrised since GS21.m has no outer break).
%
% This is NOT a port-comparison harness -- gs21_solve.py deliberately uses a
% different iteration scheme (a fused single loop). It is retained to reproduce
% the period-2 orbit finding IN MATLAB, independently of the Python: run it and
% watch |P_k - P_k-1| pin to a constant while |P_k - P_k-2| collapses to ~0.
%
% Needs the inputs py_ref.py dumps (grids, pr matrices, GH nodes, params).
% Use dims where the solve is stable: xnum must be 20 (see README).
warning('off','all');
here = fileparts(mfilename('fullpath'));
rd = @(n) dlmread(fullfile(here, [n '.csv']), ',');
run(fullfile(here, 'params.m'));
run(fullfile(here, 'loop_args.m'));       % maxouter, maxit

bgrid=rd('bgrid'); xgrid=rd('xgrid'); zgrid=rd('zgrid');
pr_mat_x=rd('pr_x'); pr_mat_z=rd('pr_z'); nodes=rd('nodes'); weights=rd('weights');
pr_mat = kron(pr_mat_z, pr_mat_x); pr_mat_re = kron(pr_mat, ones(bnum,1));
statenum = bnum*xnum*znum;
grid_val=zeros(statenum,3); grid_ind=zeros(statenum,3);
grid_val(:,1)=kron(ones(xnum*znum,1),bgrid);
grid_val(:,2)=kron(ones(znum,1),kron(xgrid,ones(bnum,1)));
grid_val(:,3)=kron(zgrid,ones(bnum*xnum,1));
grid_ind(:,1)=kron(ones(xnum*znum,1),(1:bnum)');
pi_Rmat=(exp(grid_val(:,2)+grid_val(:,3))-delta)-(1-tau).*grid_val(:,1);
def_Rmat_pre=phi*(1-delta+exp(reshape(grid_val(:,2)+grid_val(:,3),bnum,xnum*znum)));
Mmat=exp(-r-0.5*gamma_x^2-gamma_x*(xgrid'-(1-rho_x)*x_bar-rho_x*grid_val(:,2))/sigma_x);
Mmat=repmat(Mmat,1,znum);

P_up_old=zeros(statenum,1); P_down_old=zeros(statenum,1);
Q_0_old=zeros(statenum,1); Q_I_re_old=zeros(statenum,1);
prob_i_up=1; prob_i_down=1;
tol_debt=1e-3; tol_price=1e-4;
hist = zeros(statenum, maxouter);
for vfct=1:maxouter
  for vfct_debt=1:maxit
    Q_up_old   = prob_i_up.*Q_I_re_old(:)   + (1-prob_i_up).*Q_0_old;
    Q_down_old = prob_i_down.*Q_I_re_old(:) + (1-prob_i_down).*Q_0_old;
    EQ0_up = sum(((grid_val(:,1)+Q_up_old).*((P_up_old+nodes')>0) + def_Rmat_pre(:).*((P_up_old+nodes')<=0)).*(weights.'),2);
    EQ0_up = sum(Mmat.*(repmat(reshape(EQ0_up,bnum,xnum*znum),xnum*znum,1)).*pr_mat_re,2);
    EQ0_down = sum(((grid_val(:,1)+Q_down_old).*((P_down_old+nodes')>0) + def_Rmat_pre(:).*((P_down_old+nodes')<=0)).*(weights.'),2);
    EQ0_down = sum(Mmat.*repmat(reshape(EQ0_down,bnum,xnum*znum),xnum*znum,1).*pr_mat_re,2);
    Q_0 = xi*EQ0_up + (1-xi)*EQ0_down;
    Q_I = g*Q_0; Q_I_re = reshape(Q_I,bnum,xnum*znum);
    Q_I_no = interp1(bgrid,Q_I_re,bgrid/g,'spline','extrap');
    solerr = max(abs((Q_0(:)-Q_0_old(:))));
    if solerr < tol_debt
      tol_debt = tol_debt/1.2; break
    end
    Q_0_old = Q_0; Q_I_re_old = Q_I_no;   % GS21.m:239-240, skipped on break
  end
  for vfct_price=1:maxit
    prof0_up = pi_Rmat + ((1-kappa_b)*kron(reshape(Q_0,bnum,xnum*znum)',ones(bnum,1)) - Q_0);
    P0_up_R = (1+(prof0_up<=0).*kappa_e).*prof0_up;
    P0_down_R = (1+(pi_Rmat<=0).*kappa_e).*pi_Rmat;
    profI_up = pi_Rmat + ((1-kappa_b)*kron(Q_I_re',ones(bnum,1)) - Q_I_no(:));
    PI_up_R = (1+(profI_up<=0).*kappa_e).*profI_up;
    PI_down_R = (1+(pi_Rmat<=0).*kappa_e).*pi_Rmat;
    P_up_old_pert = sum(max(P_up_old+nodes',0).*(weights.'),2);
    P_down_old_pert = sum(max(P_down_old+nodes',0).*(weights.'),2);
    P_up_old_re = reshape(P_up_old_pert,bnum,xnum*znum);
    P_down_old_re = reshape(P_down_old_pert,bnum,xnum*znum);
    EPI_up=g*P_up_old_re;   EPI_up=sum(Mmat.*repmat(EPI_up,znum*xnum,1).*pr_mat_re,2);
    EPI_down=g*P_down_old_re; EPI_down=sum(Mmat.*repmat(EPI_down,znum*xnum,1).*pr_mat_re,2);
    EPI = xi*EPI_up+(1-xi)*EPI_down;
    EP0_up=P_up_old_re;     EP0_up=sum(Mmat.*repmat(EP0_up,znum*xnum,1).*pr_mat_re,2);
    EP0_down=P_down_old_re; EP0_down=sum(Mmat.*repmat(EP0_down,znum*xnum,1).*pr_mat_re,2);
    EP0 = xi*EP0_up+(1-xi)*EP0_down;
    EP0 = kron(reshape(EP0,bnum,xnum*znum)',ones(bnum,1));
    EPI = kron(reshape(EPI,bnum,xnum*znum)',ones(bnum,1));
    P0_up=P0_up_R+EP0; P0_down=P0_down_R+EP0; PI_up=PI_up_R+EPI; PI_down=PI_down_R+EPI;
    [P0_up,no_up_bprime]=max(P0_up,[],2);
    P0_down=interp1(bgrid,P0_down',bgrid,'spline','extrap')';
    P0_down=P0_down(sub2ind(size(P0_down),(1:size(P0_up,1)).',grid_ind(:,1)));
    [PI_up,I_up_bprime]=max(PI_up,[],2);
    PI_down=interp1(bgrid,PI_down',bgrid/g,'spline','extrap')';
    PI_down=PI_down(sub2ind(size(PI_down),(1:size(PI_down,1)).',grid_ind(:,1)));
    i_cut_up=min(imax,max(imin,(PI_up-P0_up)));
    i_cut_down=min(imax,max(imin,(PI_down-P0_down)));
    prob_i_up=(i_cut_up-imin)/(imax-imin);
    P_up=prob_i_up.*(PI_up-0.5*(i_cut_up+imin))+(1-prob_i_up).*P0_up;
    [z_cut_up,weight_cut_up]=update_cutoffs(P_up,zgrid,bnum,xnum,znum);
    prob_i_down=(i_cut_down-imin)/(imax-imin);
    P_down=prob_i_down.*(PI_down-0.5*(i_cut_down+imin))+(1-prob_i_down).*P0_down;
    [z_cut_down,weight_cut_down]=update_cutoffs(P_down,zgrid,bnum,xnum,znum);
    solerr=max(abs((P_down(:)-P_down_old(:))));
    if solerr<tol_price
      tol_price=tol_price/1.2; break
    end
    P_up_old=P_up; P_down_old=P_down;
  end
  hist(:,vfct) = P_up;
end

% the orbit diagnostic
printf('\n  sweep   |P_k - P_k-1|   |P_k - P_k-2|\n');
for k = 3:maxouter
  printf('  %5d   %13.4e   %13.4e\n', k, ...
         max(abs(hist(:,k)-hist(:,k-1))), max(abs(hist(:,k)-hist(:,k-2))));
end
printf('\n  A constant column 1 with column 2 -> 0 is an exact period-2 orbit.\n');
dlmwrite(fullfile(here,'ol_P_up.csv'), P_up, 'delimiter',',','precision','%.17g');
dlmwrite(fullfile(here,'ol_P_down.csv'), P_down, 'delimiter',',','precision','%.17g');
dlmwrite(fullfile(here,'ol_Q_0.csv'), Q_0, 'delimiter',',','precision','%.17g');
