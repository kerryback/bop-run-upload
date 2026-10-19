Columns: SR_max, the linear ceiling and the room are the population investor's conditional Sharpe over the evaluation window (room = best nonlinear ceiling minus the linear ceiling, and room % lin = room as a percentage of the linear ceiling); DKKM, FMR, FF, linear (the best of rank and level regressions, with and without the market) and CAPM (the estimated market line) are out-of-sample Sharpe ratios at the 360-month window; fair gap = DKKM minus the best of (FMR, FF, linear, CAPM), mean (sd) across seeds; + = seeds with a positive fair gap. A dagger marks an oracle that predates the evaluation-window ceilings.

### Baselines

The three models as published, one type each: the floor every group is read against.

| model | economy | SR_max | market % | linear ceiling | room | room % lin | DKKM | FMR | FF | linear | CAPM | fair gap | + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BGN | `bgnbase` -- baseline, no priced state beyond the rate | 0.301 | 54% | 0.250 | +0.027 | 11 | 0.238 | 0.213 | 0.208 | 0.238 | 0.162 | -0.0001 (0.006) | 6/10 |
| GS21 | `gsbase` -- baseline | 0.293 | 95% | 0.291 | +0.001 | 0 | 0.283 | 0.265 | 0.261 | 0.286 | 0.278 | -0.0028 (0.002) | 2/10 |
| KP14 | `kpbase` -- baseline, one type, no priced state | 0.229 | 35% | 0.221 | +0.004 | 2 | 0.208 | 0.193 | 0.195 | 0.209 | 0.078 | -0.0010 (0.006) | 6/10 |

### Group 1. A different distribution of shocks: a crash-only factor

Every firm's cash flows load on a factor y whose only priced shock is a rare crash (1\% a month to $-3.5$ sd, kernel weight $e^2$).

| model | economy | SR_max | market % | linear ceiling | room | room % lin | DKKM | FMR | FF | linear | CAPM | fair gap | + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BGN | `bvy1j` -- crash-only factor: e^0.035y, the same crash | 0.324 | 50% | 0.278 | +0.033 | 12 | 0.265 | 0.239 | 0.235 | 0.262 | 0.160 | +0.0003 (0.010) | 6/10 |
| GS21 | `gvyz10j` -- crash-only factor on 0.10 z y: y unpriced except the 1%/month jump to -3.5 (weight e^2) | 0.386 | 82% | 0.374 | +0.011 | 3 | 0.358 | 0.342 | 0.334 | 0.363 | 0.316 | -0.0047 (0.004) | 1/10 |
| KP14 | `vy1j` -- crash-only factor: e^0.07y, y unpriced except a 1%/month jump to -3.5 (weight e^2) | 0.336 | 21% | 0.277 | +0.012 | 4 | 0.259 | 0.241 | 0.249 | 0.265 | 0.066 | -0.0057 (0.008) | 2/10 |

### Group 2. A quadratic price of the model's own risk

The price of the baseline's aggregate risk becomes $1 + 0.25 s^2$ times its value, capped at 3, $s$ a standardized aggregate state.

| model | economy | SR_max | market % | linear ceiling | room | room % lin | DKKM | FMR | FF | linear | CAPM | fair gap | + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BGN | `bq25` -- sigma_z x min(1 + 0.25 s^2, 3), s the standardized rate, 7 bins | 0.411 | 36% | 0.346 | +0.049 | 15 | 0.339 | 0.303 | 0.293 | 0.335 | 0.149 | +0.0034 (0.015) | 4/10 |
| GS21 | `gq25` -- gamma_x x min(1 + 0.25 x^2, 3) | 0.345 | 95% | 0.343 | +0.002 | 0 | 0.334 | 0.322 | 0.313 | 0.338 | 0.326 | -0.0038 (0.002) | 0/10 |
| KP14 | `kpq` -- gamma_x x min(1 + 0.25 y^2, 3), y unpriced | 0.260 | 53% | 0.251 | +0.003 | 1 | 0.240 | 0.226 | 0.225 | 0.242 | 0.137 | -0.0021 (0.004) | 4/10 |

### Group 3a. Exposure captured by a single characteristic

A priced state y with one type: a firm's exposure varies only through one firm state (project mix, operating leverage, growth-option share).

| model | economy | SR_max | market % | linear ceiling | room | room % lin | DKKM | FMR | FF | linear | CAPM | fair gap | + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BGN | `bvy1ch` -- one type, project cash flows x e^0.035y | 0.336 | 54% | 0.297 | +0.029 | 10 | 0.283 | 0.252 | 0.255 | 0.281 | 0.181 | +0.0017 (0.007) | 7/10 |
| GS21 | `gvyz10` -- one type, output x e^0.10 z y: the product economy without the coupon | 0.854 | 50% | 0.800 | +0.043 | 5 | 0.793 | 0.806 | 0.743 | 0.809 | 0.421 | -0.0170 (0.010) | 1/10 |
| KP14 | `vy1` -- one type, cash flows x e^0.07y | 0.466 | 12% | 0.369 | +0.019 | 5 | 0.355 | 0.335 | 0.340 | 0.360 | 0.047 | -0.0050 (0.005) | 1/10 |

### Group 3b. Exposure as an interaction of characteristics

The same y with three types (BGN, KP14) or a product written into production (GS21): exposure = type or productivity $\times$ how levered the claim is.

| model | economy | SR_max | market % | linear ceiling | room | room % lin | DKKM | FMR | FF | linear | CAPM | fair gap | + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BGN | `bvych0` -- y as a 21-node chain, three types (control) | 0.576 | 37% | 0.341 | +0.055 | 16 | 0.334 | 0.285 | 0.292 | 0.326 | 0.216 | +0.0076 (0.009) | 8/10 |
| GS21 | `gvyp10` -- product exposure zb at 0.10 (control) | 0.650 | 8% | 0.447 | +0.194 | 47 | 0.550 | 0.189 | 0.333 | 0.531 | 0.018 | +0.0196 (0.022) | 9/10 |
| KP14 | `vyg25` -- three types on y, price 2.5 (control) | 0.495 | 11% | 0.346 | +0.082 | 24 | 0.350 | 0.292 | 0.320 | 0.335 | 0.044 | +0.0148 (0.009) | 10/10 |

### Group 4. Larger factor models: two priced states

Two independent priced states, three types with loading vectors (1,0), (0,1), (1,1)/$\sqrt2$, total price held at 2.5.

| model | economy | SR_max | market % | linear ceiling | room | room % lin | DKKM | FMR | FF | linear | CAPM | fair gap | + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BGN | `bvyk2` -- two priced chains, three loading vectors | 0.639 | 36% | 0.393 | +0.071 | 18 | 0.394 | 0.332 | 0.353 | 0.379 | 0.228 | +0.0152 (0.011) | 10/10 |
| GS21 | `gvypk3` -- two priced chains, three product types | 0.612 | 8% | 0.349 | +0.226 | 65 | 0.465 | 0.245 | 0.306 | 0.407 | 0.050 | +0.0563 (0.053) | 19/20 |
| KP14 | `vym2t3` -- two priced states, three loading vectors | 0.475 | 12% | 0.326 | +0.090 | 28 | 0.317 | 0.280 | 0.301 | 0.308 | 0.049 | +0.0075 (0.011) | 8/10 |

### Additional economies

Everything else that was run, by model; each is described in the appendix against its parent.

| model | economy | SR_max | market % | linear ceiling | room | room % lin | DKKM | FMR | FF | linear | CAPM | fair gap | + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BGN | `g0235f` -- regime price x0.2 / x3.5, spells 4x faster | 0.237 | 62% | 0.163 | +0.045 | 28 | 0.169 | 0.141 | 0.131 | 0.161 | 0.146 | +0.0079 (0.005) | 10/10 |
| BGN | `g0235df` -- stress 2/3 of the time, 4x faster | 0.489 | 17% | 0.432 | +0.036 | 9 | 0.423 | 0.392 | 0.351 | 0.428 | 0.077 | -0.0051 (0.007) | 2/10 |
| BGN | `bvy25` -- three types on y, level-compensated (1.2) | 0.601 | 41% | 0.398 | +0.062 | 16 | 0.399 | 0.344 | 0.367 | 0.384 | 0.248 | +0.0148 (0.016) | 9/10 |
| BGN | `bvych41` -- bvych0 on a 41-node chain | 0.565 | 38% | 0.336 | +0.052 | 15 | 0.328 | 0.279 | 0.291 | 0.320 | 0.215 | +0.0086 (0.010) | 8/10 |
| BGN | `bvyc0` -- the same without compensation | 0.475 | 41% | 0.395 | +0.066 | 18 | 0.386 | 0.321 | 0.347 | 0.376 | 0.195 | +0.0104 (0.011) | 9/10 |
| BGN | `bvyj25` -- the chain with a priced crash (1%/month to -y_max, priced e^2) | 0.536 | 38% | 0.315 | +0.054 | 17 | 0.307 | 0.261 | 0.269 | 0.301 | 0.202 | +0.0066 (0.007) | 9/10 |
| BGN | `bvysg` -- signed exposure ladder (-0.07 / 0 / +0.07) | 0.504 | 28% | 0.361 | +0.120 | 35 | 0.318 | 0.305 | 0.273 | 0.324 | 0.141 | -0.0199 (0.032) | 2/10 |
| GS21 | `g28` -- countercyclical gamma(x), clipped | 0.314 | 95% | 0.312 | +0.001 | 0 | 0.303 | 0.254 | 0.256 | 0.305 | 0.298 | -0.0024 (0.002) | 0/10 |
| GS21 | `bx7` -- two-state regime on gamma, five types on x | 0.327 | 94% | 0.317 | +0.007 | 2 | 0.309 | 0.282 | 0.271 | 0.311 | 0.306 | -0.0017 (0.002) | 0/10 |
| GS21 | `gx7` -- five types on productivity x | 0.485 | 98% | 0.474 | +0.008 | 2 | 0.473 | 0.393 | 0.383 | 0.473 | 0.473 | +0.0003 (0.001) | 5/10 |
| GS21 | `gvy1` -- one type, output x e^0.05y | 0.636 | 95% | 0.633 | +0.002 | 0 | 0.625 | 0.613 | 0.601 | 0.629 | 0.602 | -0.0036 (0.003) | 1/10 |
| GS21 | `gvy1j` -- crash-only factor: e^0.05y, the same crash | 0.379 | 96% | 0.377 | +0.001 | 0 | 0.372 | 0.353 | 0.343 | 0.374 | 0.363 | -0.0027 (0.001) | 0/10 |
| GS21 | `gvy25` -- production-level exposure to y, three types | 0.725 | 86% | 0.709 | +0.014 | 2 | 0.686 | 0.681 | 0.670 | 0.690 | 0.625 | -0.0041 (0.004) | 2/10 |
| GS21 | `gvys25` -- signed fixed-cost exposure to y | 0.619 | 69% | 0.602 | +0.014 | 2 | 0.496 | 0.490 | 0.426 | 0.499 | 0.420 | -0.0032 (0.005) | 2/6 |
| GS21 | `gvyp05` -- product zb at 0.05 | 0.425 | 60% | 0.284 | +0.128 | 45 | 0.296 | 0.255 | 0.206 | 0.277 | 0.254 | +0.0194 (0.010) | 10/10 |
| GS21 | `gvyp075` -- product at 0.075 | 0.561 | 28% | 0.261 | +0.287 | 111 | 0.428 | 0.172 | 0.188 | 0.351 | 0.149 | +0.0765 (0.030) | 10/10 |
| GS21 | `gvyp15` -- product at 0.15 | 0.744 | -25% | 0.616 | +0.119 | 22 | 0.661 | 0.327 | 0.515 | 0.583 | 0.192 | +0.0780 (0.040) | 10/10 |
| GS21 | `gvyp20` -- product at 0.20 | 0.778 | -51% | 0.729 | +0.039 | 6 | 0.701 | 0.452 | 0.643 | 0.680 | 0.407 | +0.0215 (0.024) | 9/10 |
| GS21 | `gvyp10x` -- product at 0.10 plus a curvature loading on x | 0.589 | 17% | 0.365 | +0.215 | 63 | 0.483 | 0.157 | 0.258 | 0.435 | 0.054 | +0.0486 (0.022) | 10/10 |
| GS21 | `gvyp10j` -- the product exposure with a priced crash | 0.537 | 21% | 0.328 | +0.197 | 62 | 0.440 | 0.210 | 0.240 | 0.409 | 0.072 | +0.0301 (0.020) | 10/10 |
| GS21 | `gvypk2` -- two priced chains, two product types | 0.478 | 27% | 0.226 | +0.212 | 103 | 0.305 | 0.143 | 0.156 | 0.254 | 0.105 | +0.0510 (0.056) | 9/10 |
| KP14 | `kpgam` -- gamma_x x logistic(y), 1 to 3 | 0.390 | 65% | 0.373 | +0.005 | 1 | 0.358 | 0.348 | 0.340 | 0.363 | 0.254 | -0.0053 (0.004) | 1/10 |
| KP14 | `vyx` -- three types on y, price 1.8 | 0.421 | 14% | 0.321 | +0.055 | 17 | 0.313 | 0.284 | 0.302 | 0.310 | 0.054 | +0.0009 (0.009) | 7/10 |
| KP14 | `vyj25` -- the same with a priced crash in y | 0.554 | 10% | 0.323 | +0.104 | 33 | 0.315 | 0.270 | 0.292 | 0.307 | 0.045 | +0.0080 (0.011) | 8/10 |
| KP14 | `vym3t3` -- three priced states, three types | 0.443 | 14% | 0.285 | +0.087 | 30 | 0.279 | 0.246 | 0.266 | 0.273 | 0.061 | +0.0036 (0.015) | 6/10 |
| KP14 | `vym3` -- three priced states, twenty rank-3 types | 0.418 | 15% | 0.298 | +0.056 | 19 | 0.286 | 0.259 | 0.278 | 0.287 | 0.059 | -0.0032 (0.006) | 5/10 |

### Reduced form

| configuration | SR_max | market % | linear ceiling | room | room % lin | DKKM | FMR | FF | linear | CAPM | fair gap | + |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kfh_lin_1 -- K=1 factor on observed characteristics, linear loadings | 0.598 | 8 | 0.598 | +0.000 | 0 | 0.590 | 0.584 | 0.581 | 0.595 | 0.041 | -0.0067 (0.009) | 0/5 |
| kfh_lin_4 -- K=4 | 0.592 | 8 | 0.588 | +0.003 | 0 | 0.555 | 0.578 | 0.567 | 0.575 | 0.041 | -0.0237 (0.017) | 0/5 |
| kfh_lin_16 -- K=16 | 0.385 | 13 | 0.383 | +0.001 | 0 | 0.352 | 0.357 | 0.353 | 0.365 | 0.041 | -0.0128 (0.025) | 2/5 |
| kfh_tanh_1 -- K=1, bounded (tanh) loadings on observed characteristics | 0.592 | 8 | 0.592 | +0.001 | 0 | 0.579 | 0.578 | 0.570 | 0.588 | 0.041 | -0.0105 (0.012) | 0/5 |
| kfh_tanh_4 -- K=4 | 0.584 | 9 | 0.574 | +0.007 | 1 | 0.541 | 0.564 | 0.544 | 0.560 | 0.041 | -0.0232 (0.018) | 0/5 |
| kfh_tanh_16 -- K=16 | 0.469 | 11 | 0.378 | +0.044 | 12 | 0.346 | 0.357 | 0.345 | 0.359 | 0.041 | -0.0152 (0.025) | 1/5 |
| kfh_sq_1 -- K=1, pure curvature (squares) of observed characteristics | 0.600 | 8 | 0.080 | +0.519 | 647 | 0.582 | 0.032 | 0.046 | 0.059 | 0.041 | +0.5161 (0.020) | 5/5 |
| kfh_sq_4 -- K=4 | 0.601 | 8 | 0.097 | +0.501 | 515 | 0.573 | 0.042 | 0.458 | 0.074 | 0.041 | +0.1155 (0.011) | 5/5 |
| kfh_sq_16 -- K=16 | 0.601 | 8 | 0.076 | +0.523 | 687 | 0.571 | 0.036 | 0.429 | 0.052 | 0.041 | +0.1429 (0.022) | 5/5 |
| kfo_sq_1 -- K=1, squares, directions orthogonal to size | 0.600 | 8 | 0.080 | +0.519 | 648 | 0.582 | 0.032 | 0.048 | 0.059 | 0.041 | +0.5163 (0.020) | 5/5 |
| kfo_sq_4 -- K=4 | 0.601 | 8 | 0.092 | +0.507 | 552 | 0.575 | 0.058 | 0.061 | 0.081 | 0.041 | +0.4833 (0.043) | 5/5 |
| kfo_sq_16 -- K=16 | 0.602 | 8 | 0.079 | +0.520 | 656 | 0.575 | 0.015 | 0.031 | 0.053 | 0.041 | +0.5030 (0.053) | 5/5 |
| lfh_L5_K1_n0 -- K=1 product of latent states, no observation noise | 0.598 | 8 | 0.208 | +0.389 | 187 | 0.570 | 0.156 | 0.539 | 0.175 | 0.041 | +0.0311 (0.017) | 5/5 |
| lfh_L5_K2_n0 -- K=2 | 0.598 | 8 | 0.144 | +0.453 | 314 | 0.564 | 0.111 | 0.520 | 0.115 | 0.041 | +0.0443 (0.015) | 5/5 |
| lfh_L5_K3_n0 -- K=3 | 0.598 | 8 | 0.106 | +0.490 | 464 | 0.566 | 0.059 | 0.507 | 0.084 | 0.041 | +0.0589 (0.015) | 5/5 |
| lfh_L5_K5_n0 -- K=5 | 0.598 | 8 | 0.092 | +0.502 | 549 | 0.543 | 0.048 | 0.423 | 0.064 | 0.041 | +0.1193 (0.012) | 5/5 |
| lfh_L5_K10_n0 -- K=10 | 0.598 | 8 | 0.089 | +0.503 | 568 | 0.507 | 0.047 | 0.250 | 0.067 | 0.041 | +0.2572 (0.014) | 5/5 |
| lfh_L5_K1_n1 -- K=1 product of latent states, observation noise 1.0 | 0.598 | 8 | 0.167 | +0.416 | 249 | 0.526 | 0.096 | 0.433 | 0.137 | 0.041 | +0.0930 (0.055) | 5/5 |
| lfh_L5_K2_n1 -- K=2 | 0.598 | 8 | 0.111 | +0.469 | 423 | 0.523 | 0.072 | 0.420 | 0.082 | 0.041 | +0.1025 (0.034) | 5/5 |
| lfh_L5_K3_n1 -- K=3 | 0.598 | 8 | 0.091 | +0.486 | 537 | 0.528 | 0.057 | 0.391 | 0.073 | 0.041 | +0.1371 (0.030) | 5/5 |
| lfh_L5_K5_n1 -- K=5 | 0.598 | 8 | 0.075 | +0.496 | 664 | 0.496 | 0.038 | 0.309 | 0.060 | 0.041 | +0.1866 (0.016) | 5/5 |
| lfh_L5_K10_n1 -- K=10 | 0.598 | 8 | 0.079 | +0.479 | 607 | 0.456 | 0.052 | 0.171 | 0.077 | 0.041 | +0.2854 (0.026) | 5/5 |

