% GS21.m:395-408, verbatim. Kept as a standalone function so both the
% single-pass harness and the fuzz harness call the same MATLAB source.
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
end
