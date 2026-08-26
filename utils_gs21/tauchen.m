function [z0, pr_mat_z] = tauchen(sigma, rho, multiple, znum)

sdz=sqrt(sigma^2/(1-rho^2));

lb = -multiple*sdz;
ub = multiple*sdz;

z0 = linspace(lb,ub,znum)';

gridinc = z0(2)-z0(1);

pr_mat_z = zeros(znum,znum);

for zct=1:znum
   
    zval = z0(zct);
    meanval = rho*zval;
    
    for zprimect=2:(znum-1)
   
     pr_mat_z(zct,zprimect) = normcdfbase(z0(zprimect)+gridinc/2,meanval,sigma)-...
            normcdfbase(z0(zprimect)-gridinc/2,meanval,sigma);
    
        
    end
    
     pr_mat_z(zct,1) = normcdfbase(z0(1)+gridinc/2,meanval,sigma);
     pr_mat_z(zct,znum) = 1-normcdfbase(z0(znum)-gridinc/2,meanval,sigma);
     
     pr_mat_z(zct,:) = pr_mat_z(zct,:)/sum(pr_mat_z(zct,:));
end

pr_mat_z(pr_mat_z<0) = 0;

z0 = (z0);

end

function out = normcdfbase(x,mu,sigma)

std = (x-mu)/sigma;

out = 0.5*erfc(-std/sqrt(2));

end