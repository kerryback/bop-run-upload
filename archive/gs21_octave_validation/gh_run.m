% Golub-Welsch Gauss-Hermite exactly as GS21.m:410-432 computes it, so scipy's
% roots_hermite can be checked against it. Note eig() on a full n x n matrix:
% n=20000 needs ~7 GB, so compare at moderate n and use the range-restricted
% Python check (see README) for the production node count.
warning('off','all');
here = fileparts(mfilename('fullpath'));
run(fullfile(here,'gh_args.m'));
i = (1:n-1)'; beta = sqrt(i/2);
J = diag(beta,1) + diag(beta,-1);
[V, D] = eig(J);
x = diag(D); w = V(1,:).^2 * sqrt(pi);
dlmwrite(fullfile(here,'gh_x.csv'), x(:), 'delimiter',',','precision','%.17g');
dlmwrite(fullfile(here,'gh_w.csv'), w(:), 'delimiter',',','precision','%.17g');
