#!/usr/bin/python3

import numpy as np
from deeptime.markov.tools.estimation import rate_matrix
from mfpt import MFPT_from_Generator_M2, MFPT_from_Generator_Dgeo, MFPT_from_Generator_Dpot
from mfpt import MFPT_from_Generator_Djump, MFPT_from_Generator_Poisson
from mfpt import timescales_from_Q, stationary_distribution_from_Generator
from count import countMatrix, bootstrap_trajectories
from generators import diagonalAdjustment, MLE, polySolver2, polySolver4

import warnings
warnings.filterwarnings(
    "ignore",
    module=r"deeptime(\.|$)")

def estimatorFunction(args):
    """
    For a single bootstrap iteration with given discrete trajectories and lag:
      - Bootstrap sample the discrete trajectories
      - Compute count matrix
      - Solve for rate matrix
      - Calculate FE, MFPT, and other observables
      - Return results for this single bootstrap
    """
    i, valid_trajs, block_length, lag, dt_, unit_phys, r, tang, NSTATE, First, Second, estimator, FE_, kbT, stride_k = args[:15]
    guessD0 = args[15] if len(args) > 15 else None

    if FE_ is not None:
       pi_ = np.exp(-(FE_ - FE_.min()) / kbT)
       pi_ /= pi_.sum()
    else:
       pi_ = None
    
    fe_results = {}
    mfpt_results = {}
    D_results = {}

    max_retries = 50
    attempt = 0
    Q = None

    # Set binwidth
    binwidth = 1.0 / (NSTATE-1)

    try:
        # Draw one bootstrap sample and keep it fixed if the solver is retried
        md_dtrajs_boot = bootstrap_trajectories(valid_trajs, block_length)

        # Build the count matrix
        Nmat = countMatrix(md_dtrajs_boot, lag, n_states=NSTATE, mode="sliding", stride=None)
    except Exception as e:
        print(f"Warning: Bootstrap {i} could not be constructed. Skipping. Error: {e}")
        return fe_results, mfpt_results, D_results

    while attempt < max_retries:
        try:
            attempt += 1

            # Attempt to calculate the rate matrix
            with np.errstate(over='ignore'):
                if estimator == 'KL':
                    Q = rate_matrix(Nmat, dt=lag*dt_, method='KL', pi=None, on_error='raise')
                elif estimator == 'KL-pi':
                    Q = rate_matrix(Nmat, dt=lag*dt_, method='KL', pi=pi_,  on_error='raise')
                elif estimator == 'CVE':
                    Q = rate_matrix(Nmat, dt=lag*dt_, method='CVE', pi=None, on_error='raise')
                elif estimator == 'DA':
                    Q = diagonalAdjustment(Nmat, lag*dt_)
                elif estimator == 'WA':
                    Q = weightedAdjustment(Nmat, lag*dt_)
                elif estiamtor == 'CWO':
                    Q = CWO(Nmat, lag*dt_)
                elif estiamtor == 'QOG': 
                    Q = QOG(Nmat, lag*dt_)
                elif estimator == 'PA':
                    Q = polySolver2(Nmat, lag*dt_, LogMethod='safelog')
                elif estimator == 'PA-pi':
                    Q = polySolver4(Nmat, lag*dt_, pi=pi_, LogMethod='safelog')
                elif estimator == 'MLE-pi':
                    _, Q = MLE(Nmat, FE_, binwidth, kbT, lag*dt_, lam=1e-6, D0=None, disp=False)
                else:
                    raise ValueError('Unknown Generator!')
            #Q = validateGenerator(Q)
            break 

        except Exception as e:
            if attempt == max_retries:
                print(f"Warning: Bootstrap {i} failed after {max_retries} attempts. Skipping. Error: {e}")
                return fe_results, mfpt_results, D_results
            continue
 
    #######################
    # FE/MFPT calculations
    #######################
    stat_dist = stationary_distribution_from_Generator(Q)
    if np.any(~np.isfinite(stat_dist)) or np.any(stat_dist <= 0):
        stat_dist = np.where(np.isfinite(stat_dist) & (stat_dist > 0), stat_dist, 1e-16)
        stat_dist /= stat_dist.sum()
    c1_t = -kbT * np.log(stat_dist)
    fe = c1_t - np.min(c1_t)

    fe_smooth = 0.0 
    its = timescales_from_Q(Q, k=3)
    
    mfpt0, D1, D1s, D1pot, D1pots, D1geo, D1geos = MFPT_from_Generator_Dgeo(Q, fe, first=First, second=Second, delta=1, 
                                    binwidth=binwidth, lag=lag, R=r, kbT=kbT, unit=unit_phys)
    #TODO
    fe_corr = fe

    mfpt1, mfpt5, c2, Dhops, Dhopss, Dhopspot, Dhopspots, Dhopgeo, Dhopgeos = MFPT_from_Generator_M2(Q, fe, first=First, second=Second, delta=1,
                                                binwidth=binwidth, lag=lag, R=r, pi=None, kbT=kbT, unit=unit_phys)
    #TODO
    fe_corr2 = c2

    mfpt0, D1, D1s, D1pot, D1pots, D1geo, D1geos = MFPT_from_Generator_Dgeo(Q, fe_corr, first=First, second=Second, delta=1, 
                                    binwidth=binwidth, lag=lag, R=r, kbT=kbT, unit=unit_phys)
    mfpt1, mfpt5, _, DM2, DM2s, DM2pot, DM2pots, DM2geo, DM2geos = MFPT_from_Generator_M2(Q, fe_corr2, first=First, second=Second, delta=1, 
                                                binwidth=binwidth, lag=lag, R=r, pi=None, kbT=kbT, unit=unit_phys)
    mfpt2 = MFPT_from_Generator_Poisson(Q, first=First, second=Second, delta=1, 
                                        binwidth=binwidth, R=r, lag=lag, unit=unit_phys)
    mfpt3, _, _, _, _, _, _ = MFPT_from_Generator_Djump(Q, fe_corr, first=First, second=Second, delta=1,
                                                binwidth=binwidth, lag=lag, R=r, kbT=kbT, unit=unit_phys)
    mfpt4, _, _, _, _, _, _ = MFPT_from_Generator_Dpot(Q, fe_corr, first=First, second=Second, delta=1,
                                                binwidth=binwidth, lag=lag, R=r, kbT=kbT, unit=unit_phys)

    mfpt6, _, _ , _, _, _, _ = MFPT_from_Generator_Dgeo(Q, fe_corr, first=Second, second=First, delta=1, 
                                   binwidth=binwidth, lag=lag, R=r, kbT=kbT, unit=unit_phys) 
    mfpt7, _, _, _, _, _, _, _, _ = MFPT_from_Generator_M2(Q, fe_corr2, first=Second, second=First, delta=1, 
                                            binwidth=binwidth, lag=lag, R=r, pi=None, kbT=kbT, unit=unit_phys)
    mfpt8 = MFPT_from_Generator_Poisson(Q, first=Second, second=First, delta=1, binwidth=binwidth, R=r, lag=lag, unit=unit_phys)
    
    fe_results[(r, tang, lag)] = (fe, fe_smooth, fe_corr, fe_corr2)
    mfpt_results[(r, tang, lag)] = (float(mfpt0), float(mfpt1), float(mfpt2), float(mfpt3), float(mfpt4), float(mfpt5),
                                    float(mfpt6), float(mfpt7), float(mfpt8),
                                    float(its[0]), float(its[1]), float(its[2]))
    D_results[(r, tang, lag)] = (D1geo, D1geos, D1, D1s, D1pot, D1pots, DM2geo, DM2geos, DM2pot, DM2pots)
    
    return fe_results, mfpt_results, D_results
