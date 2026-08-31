% Driver for the tauchen.py vs tauchen.m comparison. Needs tauchen.m on the
% path: run from utils_gs21/ or copy tauchen.m in beside this file.
warning('off','all');
here = fileparts(mfilename('fullpath'));
run(fullfile(here,'tch_args.m'));
[g, pr] = tauchen(sigma, rho, mnstdev, num);
dlmwrite(fullfile(here,'tch_g.csv'), g, 'delimiter',',','precision','%.17g');
dlmwrite(fullfile(here,'tch_pr.csv'), pr, 'delimiter',',','precision','%.17g');
