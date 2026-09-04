import numpy as np
import pandas as pd
import os
from scipy import interpolate
from config import (
    KP14_BURNIN as burnin, KP14_DT as dt,
    KP14_MU_X as mu_x, KP14_MU_Z as mu_z,
    KP14_SIGMA_X as sigma_x, KP14_SIGMA_Z as sigma_z,
    KP14_THETA_EPS as theta_eps, KP14_SIGMA_EPS as sigma_eps,
    KP14_THETA_U as theta_u, KP14_SIGMA_U as sigma_u,
    KP14_DELTA as delta, KP14_MU_LAMBDA as mu_lambda,
    KP14_SIGMA_LAMBDA as sigma_lambda,
    # 2026-09-04: local mu_H/mu_L are EXIT rates (rate of LEAVING that state).
    # config's KP14_MU_H/MU_L are ENTRY rates, so they map across. See
    # docs/kp14_regime_labels.md and the note at config.py:KP14_PROB_H.
    KP14_EXIT_H as mu_H, KP14_EXIT_L as mu_L,
    KP14_LAMBDA_H as lambda_H, KP14_LAMBDA_L as lambda_L,
    KP14_R as r, KP14_GAMMA_X as gamma_x, KP14_GAMMA_Z as gamma_z,
    KP14_ALPHA as alpha, KP14_PROB_H as prob_H,
    KP14_A_0 as A_0, KP14_A_1 as A_1, KP14_A_2 as A_2, KP14_A_3 as A_3,
    KP14_RHO as rho, KP14_C as C
)

# Define A function (from parameters_kp14.py)
A = lambda ep, u: (A_0 + (ep - 1) * A_1 + (u - 1) * A_2 + (ep - 1) * (u - 1) * A_3)

# Analytic Hansen-Jagannathan bound: sd(M)/E[M] for the lognormal SDF
# M = exp(-r dt - 0.5|g|^2 dt - g'dW). This is the maximum Sharpe ratio any
# portfolio can attain over one dt, and it is constant. See the long note at the
# max_sr assignment for why the estimated version was withdrawn.
MAX_SHARPE = np.sqrt(np.exp((gamma_x**2 + gamma_z**2) * dt) - 1.0)

import scipy.special
from scipy.sparse import csr_matrix, diags, kron, vstack

# sdf_compute( ) computes second moment of returns after loading integrals computed in integ_kp14

### STOPPED FOR NOW -- should check everything carefully again after checking overleaf
## the top level code and sdf_loop run currently

# Get path to solution files relative to this module
_SOLFILES_DIR = os.path.join(os.path.dirname(__file__), 'KP14_solfiles')

# Provenance check: these solution files are committed artifacts produced
# offline by utils_kp14/regen_solfiles.py. This refuses to run if they do not
# match the code and parameters that claim to produce them -- config.py drift, a
# producer edited without regenerating, or a stale G_func.csv paired with fresh
# integrals. All three have happened. Regenerate with:
#     python utils_kp14/regen_solfiles.py
# See utils/solfile_stamp.py and kp14_crash_20260826.md.
from .solfile_spec import verify as _verify_solfiles
_verify_solfiles(mode='error')

# read in G functions estimated in kp14_fd.py
# recall they don't include lambda_ft, which varies across firms and time
G_in = pd.read_csv(os.path.join(_SOLFILES_DIR, 'G_func.csv'))
eps_grid = G_in.eps.values
# Interpolation order, set 2026-08-31: cubic, NOT the interp1d default (linear).
#
# These are tabulated on a 1000-point uniform eps grid (h = 4.995e-3) and
# evaluated at FIRM-SPECIFIC eps, so every firm picks up an independent error.
# Linear was costing real accuracy on the three (eps-1)*f(eps) integrals, which
# cross zero at eps ~ 1 -- exactly where the CIR distribution for eps
# concentrates -- and which are the ones carrying the risk premium:
#
#     ep_A_mod_lst    RMS relative linear-interp error 7.8e-03 (max 3.7e-01)
#     ep_G_up_lst                                      2.5e-03
#     ep_G_down_lst                                    2.0e-03
#     the other eight                           1.6e-06 to 8.1e-05
#
# The driver is the curvature-to-level ratio |f''|max/|f|: 133, 35 and 18 for
# those three against 0.8-64 for the rest. Linear error is h^2*f''/8, so
# dividing by a level that passes through zero is what inflates it.
#
# Consequence: mu carried a diffuse per-firm error of ~2-4e-4 (1.5-2.8% of its
# mean) with no factor structure. Sigma^-1 at N=1000 levered that into
# sqrt(mu' Sigma^-1 mu) exceeding the analytic Hansen-Jagannathan bound in 770
# of 3,960 panel-months. See the note at max_sr in sdf_compute_kp14.py.
#
# Cubic, not pchip: all eleven integrals are strictly MONOTONE with zero turning
# points over the eps band, so there is no shape hazard for pchip to guard
# against, and measured cubic overshoot outside the data range is exactly zero.
# Held-out tests put cubic 2-6x ahead of pchip at the shipped spacing and
# 40-130x ahead as spacing coarsens; akima is worse than pchip throughout.
G_up = interpolate.interp1d(eps_grid, G_in.G_up.values, kind='cubic', fill_value="extrapolate")
G_down = interpolate.interp1d(eps_grid, G_in.G_down.values, kind='cubic', fill_value="extrapolate")

A_mod = lambda eps: A(eps, 1)**(1/(1-alpha))

# Load the integrals file
integ_data = np.load(os.path.join(_SOLFILES_DIR, 'integ_results.npz'))

# Access each array
A_mod_lst = integ_data["A_mod_lst"]
G_up_lst = integ_data["G_up_lst"]
G_down_lst = integ_data["G_down_lst"]
ep_A_mod_lst = integ_data["ep_A_mod_lst"]
ep_G_up_lst = integ_data["ep_G_up_lst"]
ep_G_down_lst = integ_data["ep_G_down_lst"]
A_mod2_lst = integ_data["A_mod2_lst"]
G_up2_lst = integ_data["G_up2_lst"]
G_down2_lst = integ_data["G_down2_lst"]
A_mod_G_up_lst = integ_data["A_mod_G_up_lst"]
A_mod_G_down_lst = integ_data["A_mod_G_down_lst"]

# define approximated integrals
Et_A_mod = interpolate.interp1d(eps_grid, A_mod_lst, kind='cubic', fill_value="extrapolate")
Et_G_up = interpolate.interp1d(eps_grid, G_up_lst, kind='cubic', fill_value="extrapolate") # again - be sure to adjust any G term by lambda_f
Et_G_down = interpolate.interp1d(eps_grid, G_down_lst, kind='cubic', fill_value="extrapolate")
Et_ep_A_mod = interpolate.interp1d(eps_grid, ep_A_mod_lst, kind='cubic', fill_value="extrapolate")
Et_ep_G_up = interpolate.interp1d(eps_grid, ep_G_up_lst, kind='cubic', fill_value="extrapolate")
Et_ep_G_down = interpolate.interp1d(eps_grid, ep_G_down_lst, kind='cubic', fill_value="extrapolate")
Et_A_mod_sq = interpolate.interp1d(eps_grid, A_mod2_lst, kind='cubic', fill_value="extrapolate")
Et_G_up_sq = interpolate.interp1d(eps_grid, G_up2_lst, kind='cubic', fill_value="extrapolate")
Et_G_down_sq = interpolate.interp1d(eps_grid, G_down2_lst, kind='cubic', fill_value="extrapolate")
Et_A_mod_G_up = interpolate.interp1d(eps_grid, A_mod_G_up_lst, kind='cubic', fill_value="extrapolate")
Et_A_mod_G_down = interpolate.interp1d(eps_grid, A_mod_G_down_lst, kind='cubic', fill_value="extrapolate")

# compute more integrals analytically
Et_epsm1 = lambda eps: np.exp(-theta_eps*dt)*(eps - 1)
Et_um1 = lambda u: np.exp(-theta_u*dt)*(u - 1)
Et_epsm1_sq = lambda eps: eps*sigma_eps**2/theta_eps*(np.exp( - theta_eps*dt) - np.exp(-2*theta_eps*dt)) + sigma_eps**2/(2*theta_eps) *(1 - np.exp( - theta_eps*dt))**2 + Et_epsm1(eps)**2
Et_um1_sq = lambda u: u*sigma_u**2/theta_u*(np.exp( - theta_u*dt) - np.exp(-2*theta_u*dt)) + sigma_u**2/(2*theta_u) *(1 - np.exp( - theta_u*dt))**2 + Et_um1(u)**2

# A_0, A_1, A_2, A_3 are stored in parameters.kp14.py
Et_A_sq = lambda eps, u: (A_0**2 + A_1**2*Et_epsm1_sq(eps) + A_2**2 * Et_um1_sq(u) + A_3**2 * Et_epsm1_sq(eps) * Et_um1_sq(u) + 
        2*A_0*A_1*Et_epsm1(eps) + 2*A_0*A_2*Et_um1(u) + 2*A_0*A_3*Et_epsm1(eps)*Et_um1(u) + 
        2*A_1*A_2*Et_epsm1(eps)*Et_um1(u) + 2*A_1*A_3*Et_epsm1_sq(eps)*Et_um1(u) + 2*A_2*A_3*Et_epsm1(eps)*Et_um1_sq(u))

Et_A1_A2 = lambda eps, u1, u2: (A_0**2 + A_1**2*Et_epsm1_sq(eps) + A_2**2 * Et_um1(u1) * Et_um1(u2) + A_3**2 * Et_epsm1_sq(eps) * Et_um1(u1) * Et_um1(u2) + 
        2*A_0*A_1*Et_epsm1(eps) + A_0*A_2*(Et_um1(u1) + Et_um1(u2)) + A_0*A_3*Et_epsm1(eps)*(Et_um1(u1) + Et_um1(u2)) + 
        A_1*A_2*Et_epsm1(eps)*(Et_um1(u1) + Et_um1(u2)) + A_1*A_3*Et_epsm1_sq(eps)*(Et_um1(u1) + Et_um1(u2)) + 2*A_2*A_3*Et_epsm1(eps)*Et_um1(u1) * Et_um1(u2))

Et_A_mod_A = lambda eps, u: (A_0 + A_2 * Et_um1(u))*Et_A_mod(eps) + (A_1 + A_3 * Et_um1(u))* Et_ep_A_mod(eps)

Et_A_G_up = lambda eps, u: (A_0 + A_2 * Et_um1(u))*Et_G_up(eps) + (A_1 + A_3 * Et_um1(u))* Et_ep_G_up(eps)
Et_A_G_down = lambda eps, u: (A_0 + A_2 * Et_um1(u))*Et_G_down(eps) + (A_1 + A_3 * Et_um1(u))* Et_ep_G_down(eps)

# Covariance computations
def sdf_compute(N, T, arr_tuple):
    # NOTE: this must match the arr_tuple that generate_panel.py SAVES, which is the
    # 16-array version -- generate_panel.py strips book and op_cashflow for kp14 before
    # writing <panel_id>_arr/. Do not add them back here (that was commit 005a63e0, which
    # crashed every kp14 job with "expected 18, got 16"; see also ada933a).
    K, x, z, eps, uj, chi, rate, high, Et_G, EtA, alph, Et_z_alph, price, ret, eret, lambda_f = arr_tuple
    part1 = (C*rate*dt*Et_A_mod(eps) + Et_G)

    # MEMORY OPTIMIZATION: part2 is now computed on-demand inside sdf_loop()
    # Previous version pre-computed: part2 = (1 - delta*dt)*np.sum(chi*EtA*K**alpha, axis = 0)
    # This created intermediate K**alpha array with shape (921, 921, 1000) = 6.32 GB
    # Combined with arr_tuple (27.24 GB), total memory requirement was 33.56 GB
    # This exceeded available memory and caused numpy._core._exceptions._ArrayMemoryError
    #
    # New approach computes part2[t,:] on-demand for each time t inside the loop:
    # - Only creates small slices K[:t+1, t, :]**alpha with max shape (921, 1000) = 0.007 GB
    # - Reduces total operations: ~359M vs ~849M (only computes needed time periods)
    # - Faster overall due to better cache locality and progressive sizing

    Et_z_alph2 = z**(2*alph)*np.exp(2*alph*mu_z*dt + alph*(3*alpha-1)/(1 - alpha)*sigma_z**2*dt)
    Et_x2 = x**2*np.exp(2*mu_x*dt + sigma_x**2*dt)

    def sdf_loop(t, iter): # compute ER at date t+1 (to t+2)
        t = t +1 # to match BGN timing
        ER = np.zeros((N+1, N+1))

        # Handle both Sparse3D (new format) and dense numpy arrays (backwards compat)
        # K, uj, chi use column slices: arr[:t+1, t, :]
        if hasattr(K, 'get_col_slice_dense'):
            K_slice = K.get_col_slice_dense(t, t + 1)
        else:
            K_slice = K[:t + 1, t, :]

        if hasattr(chi, 'get_col_slice_dense'):
            chi_slice = chi.get_col_slice_dense(t, t + 1)
        else:
            chi_slice = chi[:t+1, t, :]

        if hasattr(uj, 'get_col_slice_dense'):
            uj_slice = uj.get_col_slice_dense(t, t + 1)
        else:
            uj_slice = uj[:t+1, t, :]

        # start with f1 != f2
        Ktalpha = K_slice**alpha
        Ktalpha_sp = csr_matrix(Ktalpha)
        col = Ktalpha_sp.multiply(csr_matrix(EtA[:t + 1, t, :])) # recall from panel_function_kp14 - K = 0 if chi = 0
        result = kron(col, col)
        term1 = (1 - delta*dt)**2*result.sum(axis=0).reshape(N, N).A

        # Compute part2 on-demand for this time t only (memory optimization)
        # Uses same Ktalpha already computed above, so no redundant computation
        part2_t = (1 - delta*dt)*np.sum(chi_slice*EtA[:t+1, t, :]*Ktalpha, axis=0)
        term2 = Et_z_alph[t]*np.outer(part1[t,:], part2_t)
        term2 = term2 + term2.T

        term3 = Et_z_alph2[t]*np.outer(part1[t,:], part1[t,:])

        ''' test calculations on single f1, f2 pair - they match
        f1 = 2
        t11 = np.sum(Ktalpha[:, f1]*EtA[:t+1, t, f1])
        t12 = C*rate[t, f1]*dt*Et_A_mod(eps[t, f1]) + Et_G[t, f1]
        f2 = 3
        t21 = np.sum(Ktalpha[:, f2]*EtA[:t+1, t, f2])
        t22 = C*rate[t, f2]*dt*Et_A_mod(eps[t, f2]) + Et_G[t, f2]

        tot23 = ((1 - delta*dt)**2*t11*t21 + 
                 Et_z_alph[t]*(t21*(1 - delta*dt)*t12 + t11*(1 - delta*dt)*t22) +
                 Et_z_alph2[t]*t12*t22)
        ''' 

        # adjust for ignored cashflow terms
        ujt = uj_slice
        cf = np.sum(eps[t, :]*ujt*x[t]*Ktalpha*dt, axis = 0)
        term4 = np.outer(cf/price[t,:], 1+eret[t,:])
        term4 = term4 + term4.T  - np.outer(cf/price[t,:], cf/price[t,:]) # last term is double counted in first two, hence subtract

        ER[1:, 1:] = (term1+ term2 + term3)*Et_x2[t]/np.outer(price[t,:], price[t,:]) + term4

        # adjust for diagonal terms (f1 = f2)
        eps_rep = np.repeat(eps[t, :].reshape((1, N)), t+1, axis = 0)
        term1_diag = ((1 - delta*dt)*Ktalpha**2* Et_A_sq(eps_rep, uj_slice)).sum(axis = 0)

        row_indices = np.repeat(np.arange(t + 1), t + 1)  # length (t+1)^2
        Ktalpha_repeated = vstack([Ktalpha_sp[i, :] for i in row_indices])
        Ktalpha_tiled = vstack([Ktalpha_sp for _ in range(t + 1)])
        temp1 = Ktalpha_tiled.multiply(Ktalpha_repeated)
        
        ones_copy = temp1.copy()
        ones_copy.data[:] = 1
        eps_rep = ones_copy.multiply(eps[t,:].reshape((1, N)))

        ujt_one = Ktalpha_sp.copy()
        ujt_one.data[:] = 1
        ujt_sp = ujt_one.multiply(csr_matrix(ujt))
        ujt_repeated = ones_copy.multiply(vstack([ujt_sp[i, :] for i in row_indices]))
        ujt_tiled = ones_copy.multiply(vstack([ujt_sp for _ in range(t + 1)]))
        temp2 = temp1.copy()
        temp2.data = Et_A1_A2(eps_rep.data, ujt_repeated.data, ujt_tiled.data)
        
        temp3 = temp1.multiply(temp2)
        rows_to_zero = np.array([i*(t+1) + i for i in range(t+1)])  # diagonal rows to zero
        for k in rows_to_zero:
            start = temp3.indptr[k]
            end = temp3.indptr[k+1]
            temp3.data[start:end] = 0 
        temp3.eliminate_zeros()

        term2_diag = (1 - delta*dt)**2 * temp3.sum(axis=0).A.reshape(-1)

        #temp1 = np.tile(Ktalpha, (t+1, 1))*np.repeat(Ktalpha, t+1, axis = 0)  
        #temp2 = Et_A1_A2(np.repeat(eps[t,:].reshape((1, N)), (t+1)**2, axis = 0), np.tile(ujt, (t+1, 1)),np.repeat(ujt, t+1, axis = 0))
        #temp3 = temp1*temp2
        #for i in range(t+1):
        #    temp3[i*(t+1) + i, :]  = 0 # zero out same-project terms
        #term2_diag = (1 - delta*dt)**2*temp3.sum(axis = 0)
    
        eps_rep = np.repeat(eps[t, :].reshape((1, N)), t+1, axis = 0)
        term3_diag = 2*Et_z_alph[t]*C*rate[t,:]*dt*(1 - delta*dt)*(Et_A_mod_A(eps_rep, ujt)*Ktalpha).sum(axis = 0)

        Et_A_G = ((high[t,:] == 0)*lambda_f*((1 - mu_L*dt)* Et_A_G_down(eps_rep, ujt) + mu_L*dt * Et_A_G_up(eps_rep, ujt)) + 
            (high[t,:] == 1)*lambda_f*((1 - mu_H*dt)* Et_A_G_up(eps_rep, ujt) + mu_H*dt * Et_A_G_down(eps_rep, ujt)))

        term4_diag = 2*Et_z_alph[t]*(1 - delta*dt)*(Et_A_G*Ktalpha).sum(axis = 0)

        term5_diag = Et_z_alph2[t]*C**2*rate[t,:]*dt*Et_A_mod_sq(eps[t,:])

        Et_G_sq = ((high[t,:] == 0)*lambda_f**2*((1 - mu_L*dt)* Et_G_down_sq(eps[t,:]) + mu_L*dt  *Et_G_up_sq(eps[t, :])) + 
            (high[t,:] == 1)*lambda_f**2*((1 - mu_H*dt)* Et_G_up_sq(eps[t, :]) + mu_H*dt * Et_G_down_sq(eps[t, :])))
        term6_diag = Et_z_alph2[t]*Et_G_sq

        Et_lambdt_A_G = ((high[t,:] == 0)*lambda_f*lambda_L*dt*lambda_f*((1 - mu_L*dt)*Et_A_mod_G_down(eps[t,:]) + mu_L*dt * Et_A_mod_G_up(eps[t, :])) + 
            (high[t,:] == 1)*lambda_f*lambda_H*dt*lambda_f*((1 - mu_H*dt) * Et_A_mod_G_up(eps[t, :]) + mu_H*dt * Et_A_mod_G_down(eps[t, :])))
        term7_diag = Et_z_alph2[t]*2*C*Et_lambdt_A_G

        # adjust for dropped cash flow term on the diagonal
        term8_diag = 2*cf/price[t,:]*(1+eret[t,:])  - (cf/price[t,:])**2
        term_diag = (term1_diag + term2_diag + term3_diag + term4_diag + term5_diag + term6_diag + term7_diag) * Et_x2[t]/price[t,:]**2 + term8_diag

        ER[np.arange(1,N+1), np.arange(1,N+1)] = term_diag

        ''' test calculations on single f1 value - they match
        f1 = 2
        t1 = (1 - delta*dt)*np.sum(Ktalpha[:, f1]**2*Et_A_sq(eps[t, f1], ujt[:, f1]))
        
        temp = np.outer(Ktalpha[:, f1], Ktalpha[:, f1])
        Et_cross = Et_A1_A2(eps[t, f1], ujt[:, f1].reshape((t+1, 1)), ujt[:, f1].reshape((1, t+1)))
        tempp = temp*Et_cross
        tempp[range(t+1), range(t+1)] = 0
        t2 = (1 - delta*dt)**2*np.sum(tempp)


        t3 = C*rate[t, f1]*dt*(1 - delta*dt)*np.sum(Ktalpha[:, f1]*Et_A_mod_A(eps[t, f1], ujt[:, f1]))
        Et_A_G_f1 = ((high[t,f1] == 0)*lambda_f[f1]*((1 - mu_L*dt)* Et_A_G_down(eps[t, f1],ujt[:, f1]) + mu_L*dt * Et_A_G_up(eps[t, f1],ujt[:, f1])) + 
            (high[t,f1] == 1)*lambda_f[f1]*((1 - mu_H*dt)* Et_A_G_up(eps[t, f1],ujt[:, f1]) + mu_H*dt * Et_A_G_down(eps[t, f1],ujt[:, f1])))
        t4 = (1 - delta*dt)*np.sum(Ktalpha[:, f1]*Et_A_G_f1)
        tmiddle = 2*Et_z_alph[t]*(t3 + t4)

        t5 = C**2*rate[t, f1]*dt*Et_A_mod_sq(eps[t,f1])
        t6 = Et_G_sq[f1]
        t7 = 2*C*rate[t, f1]*dt*((high[t,f1] == 0)*lambda_f[f1]*((1 - mu_L*dt)*Et_A_mod_G_down(eps[t,f1]) + mu_L*dt * Et_A_mod_G_up(eps[t, f1])) + 
            (high[t,f1] == 1)*lambda_f[f1]*((1 - mu_H*dt) * Et_A_mod_G_up(eps[t, f1]) + mu_H*dt * Et_A_mod_G_down(eps[t, f1])))
        tend = Et_z_alph2[t]*(t5+t6+t7)

        '''

        # store entries when at least one of assets in risk free
        ER[0, 0] = np.exp(2 * r * dt)
        ER[0, 1:] = (eret[t, :] + 1) * np.exp(r * dt)
        ER[1:, 0] = (eret[t, :] + 1) * np.exp(r * dt)

        # SDF portfolio and returns
        
        # Exclude zero-capital firms (no live projects -> book = x/z*K.sum = 0,
        # i.e. bm=0) from the SDF solve. Their conditional return second-moment is
        # degenerate, so the full ER is singular; scipy's assume_a="pos" solve would
        # silently return garbage (100% portfolio weight on these firms, max_sr ~1e57)
        # without raising. They are not in the investable universe (the panel drops
        # book<=0 firms downstream), so they get zero SDF weight. See diag_singular.py.
        # book = x/z * K.sum(axis=0) with x, z > 0 (both GBM), so book[t,:] > 0 is
        # exactly K_slice.sum(axis=0) > 0. Use K_slice: book is not in the saved arr_tuple.
        keep = K_slice.sum(axis=0) > 0                # risky firms with capital
        idx = np.concatenate(([0], 1 + np.flatnonzero(keep)))  # always keep risk-free (col 0)

        # invert ER (on the non-degenerate sub-universe)
        port = np.zeros(N + 1)
        try:
            port[idx] = scipy.linalg.solve(
                ER[np.ix_(idx, idx)], np.ones((len(idx), 1)), assume_a="pos"
            ).reshape(-1)
        except Exception as e:
            print(f"An error occurred: {e}. Perturbing ER.")
            ER_sub = ER[np.ix_(idx, idx)] + np.eye(len(idx)) * 1e-6
            try:
                port[idx] = scipy.linalg.solve(ER_sub, np.ones((len(idx), 1))).reshape(-1)
            except Exception as e:
                print(f"Second attempt failed: {e}. Using fallback value for port.")
                port = np.full((N+1, ), np.nan)

        port /= port.sum()
        sdf_ret = -(port[1:] * (1 + ret[t, :] - np.exp(r*dt))).sum()

        # conditional variance of risky assets
        cond_var = ER[1:, 1:] - np.outer(1 + eret[t, :], 1 + eret[t, :])  

        # ------------------------------------------------------------------
        # max_sr is now the ANALYTIC Hansen-Jagannathan bound, not an estimate.
        #
        # The SDF here is exogenous and lognormal,
        #     M = exp(-r dt - 0.5|g|^2 dt - g'dW),   g = (gamma_x, gamma_z),
        # so E[M] = exp(-r dt) and sd(M)/E[M] = sqrt(exp(|g|^2 dt) - 1) exactly.
        # That is the maximum Sharpe ratio ANY portfolio can attain, it is a
        # constant (the prices of risk do not vary with the state), and it costs
        # nothing to evaluate. The estimated version below inverted a 1001x1001
        # near-singular second-moment matrix to recover a number we can write
        # down in closed form.
        #
        # WHY IT WAS REPLACED (measured 2026-08-31 on the 11-panel Phoenix run,
        # 3,960 panel-months). The estimate VIOLATED the bound it was estimating
        # in 770 of 3,960 months (19.4%), by up to 6.7%. Per panel the violation
        # rate ranged from 0/360 (panels 1, 5, 6) to 266/360 (panel 3, whose mean
        # was 101.3% of the bound). A conditional Sharpe cannot exceed the HJ
        # bound, so the estimator was the problem, not the economics.
        #
        # It was not the portfolio solve: computing sqrt(mu' Sigma^-1 mu)
        # directly reproduces max_sr to <1%, and both exceed the bound. It was
        # not a broken covariance: cond_var is PD everywhere (eig_min ~1e-4). It
        # was not conditioning: corr(max_sr, log cond) = +0.26 only.
        #
        # It is error accumulation in near-idiosyncratic directions. Scaling the
        # statistic in the number of assets n shows a CONSTANT per-asset
        # increment in SR^2 at large n -- 4.6e-5 for panel 3, 1.2e-5 for panel 1
        # -- where an exact factor model requires exactly zero, since mu lies in
        # the span of the loadings and those directions carry no premium. Backing
        # it out, the implied inconsistency between mu and Sigma is only
        # eps ~ 2-4e-4, i.e. 1.5-2.8% of mean mu. Sigma^-1 weights the
        # tiny-variance directions so heavily that a sub-percent inconsistency in
        # mu is levered into a 5% overshoot, and it grows with N.
        #
        # So the old max_sr was a noisy, upward-biased, panel-dependent estimate
        # of a constant. Anything that used it as a benchmark -- e.g. "fraction
        # of maximum Sharpe attained" in the DKKM-vs-FF/FM comparison -- was
        # biased DOWN by 0-7%, inconsistently across panels, which is the same
        # order as the effects being measured.
        #
        # The underlying mu/Sigma inconsistency is real and still open; it is
        # just no longer amplified into this statistic. cond_var is returned
        # unchanged, so downstream users of it are unaffected by this edit.
        #
        # Previous estimated version, kept for reference:
        #     max_sr = -(port[1:] * (1 + eret[t, :] - np.exp(r*dt))).sum() / np.sqrt(
        #         port[1:] @ (cond_var @ port[1:])
        #     )
        # ------------------------------------------------------------------
        max_sr = MAX_SHARPE

        return (
            sdf_ret,
            max_sr,
            1 + eret[t, :] - np.exp(r*dt),
            cond_var,
        )

    return sdf_loop
