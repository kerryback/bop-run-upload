% Driver for the update_cutoffs fuzz: read one random case, run GS21.m's
% version, write the result. fuzz_cutoffs.py calls this in a loop.
warning('off','all');
here = fileparts(mfilename('fullpath'));
run(fullfile(here,'fz_dims.m'));
P = dlmread(fullfile(here,'fz_P.csv'), ',');
zgrid = dlmread(fullfile(here,'fz_zgrid.csv'), ',');
[z_cut, weight_cut] = update_cutoffs(P, zgrid, bnum, xnum, znum);
dlmwrite(fullfile(here,'fz_z.csv'), z_cut, 'delimiter',',','precision','%.17g');
dlmwrite(fullfile(here,'fz_w.csv'), weight_cut, 'delimiter',',','precision','%.17g');
