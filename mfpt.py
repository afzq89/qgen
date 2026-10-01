#!/usr/bin/python3 
import numpy as np
import warnings
from scipy.signal import savgol_filter
from generators import writeMatrix
from scipy.linalg import eig

tiny = 1e-12


def positiveD(D, tol=1e-16):
    """
    Replaces any value smaller than tol in D with the tol.
    We know that D cannot be non-positive, but what about
    very small numbers? Very small numbers means a huge MFPT!
    """
    seg = np.asarray(D, dtype=float).copy()
    seg[seg < tol] = tol

    return seg

# Smoothing in log-space
def smoother(arr):
    arr = np.asarray(arr, dtype=float).copy()

    positive = arr[np.isfinite(arr) & (arr > 0)]
    if len(positive) < 4:
        return positiveD(arr)

    floor = np.median(positive) * 1e-8
    good = np.isfinite(arr) & (arr > floor)

    x = np.arange(len(arr))
    log_arr = np.interp(x, x[good], np.log(arr[good]))

    window = min(7, len(arr) if len(arr) % 2 else len(arr) - 1)
    if window < 5:
        return np.exp(log_arr)

    return np.exp(savgol_filter(log_arr, window, 2, mode="interp"))

def calcD(generatormatrix, binswidth=1, beta=None, potentialFunction=None, tol=1e-16):
    R = np.asarray(generatormatrix, dtype=float)
    N = R.shape[0]

    # 1. KINETIC ESTIMATOR (Second-Moment for Dense Matrices)
    # Calculates D directly at the nodes, capturing all long-range flux
    # This is D^jump
    Dnode = np.zeros(N)
    for i in range(N):
        sec_moment = 0.0
        for j in range(N):
            if i != j:
                dist_sq = ((j - i) * binswidth) ** 2
                sec_moment += R[i, j] * dist_sq
        Dnode[i] = 0.5 * sec_moment

    # Fix Boundary Artifacts via Linear Extrapolation:
    # At edges, the second moment halves artificially because jumps outside 
    # the domain are impossible. We extrapolate the boundary from the interior.
    # FIXME: I am unsure about these, so let's remove it.
    # Updated: At the i=0 i=last, we only have one term (instead of two, e.g. dist_sq / 2 * R_0,1);
    # therefore we need to consider some extrapolations! We have some options and it really depends
    # how R is. 
    # The safest option is a zero-slope boundary assumption
    Dnode[0]  = Dnode[1]
    Dnode[-1] = Dnode[-2]
    # Another option is to assume D varies approximately linearly near the endpoint: D0-D1~D1-D2=> D0=2D1-D2
    #Dnode[0]  = max(2.0 * Dnode[1]  - Dnode[2], tiny)
    #Dnode[-1] = max(2.0 * Dnode[-2] - Dnode[-3], tiny)
    
    # Safeguard against negative extrapolated values
    #Dnode[0]  = max(Dnode[0], tol)
    #Dnode[-1] = max(Dnode[-1], tol)

    # 2. GOEMETRIC ESTIMATOR
    # This is D^geo
    # nearest-neighbor off-diagonals
    # FIXME: I am unsure about these, so let's remove them.
    #lower = np.clip(np.diag(R, -1), tol, None)   # Q_{i+1, i}
    #upper = np.clip(np.diag(R,  1), tol, None)   # Q_{i, i+1}
    lower = np.diag(R, -1)
    upper = np.diag(R, 1)

    # Do we very tiny Qs?
    #unresolved = (np.abs(upper) < tiny) | (np.abs(lower) < tiny)

    Dedge_geom = (binswidth**2) * np.sqrt(lower * upper)
    #Dedge_geom[unresolved] = np.nan
    #if unresolved.any():
    #   print(f"There are very small Q! The Dgeo may not be accurate!")

    # Map edge-based Dgeo to nodes using a linear extrapolation
    Dgeo = np.zeros(N)
    Dgeo[1:-1] = 0.5 * (Dedge_geom[:-1] + Dedge_geom[1:])
    
    Dgeo[0] = Dedge_geom[0]
    Dgeo[-1] = Dedge_geom[-1]

    # Extrapolating half-step edges to integer nodes
    # D_0 = D_0.5 - 0.5 * (D_1.5 - D_0.5) = 1.5*D_0.5 - 0.5*D_1.5
    #Dgeo[0]  = max(1.5 * Dedge_geom[0]  - 0.5 * Dedge_geom[1],  1.5 * Dedge_geom[1]  - 0.5 * Dedge_geom[0])
    #Dgeo[-1] = max(1.5 * Dedge_geom[-1] - 0.5 * Dedge_geom[-2], 1.5 * Dedge_geom[-1]  - 0.5 * Dedge_geom[-2])
    #print('Dgeo: ', Dgeo)

    #3.POTENTIAL-BASED ESTIMATOR
    # This is D^pot    
    if potentialFunction is not None and beta is not None:
        U = np.asarray(potentialFunction, dtype=float)
        dU = U[1:] - U[:-1]
        Dpot_nn = (binswidth**2) * lower * np.sqrt(np.exp(-beta * dU))
        
        # Map edge-based Dpot to nodes using a linear extrapolation
        Dpot = np.zeros(N)
        Dpot[1:-1] = 0.5 * (Dpot_nn[:-1] + Dpot_nn[1:])

        Dpot[0] = Dpot_nn[0]
        Dpot[-1] = Dpot_nn[-1]
        
        # Extrapolating half-step edges to integer nodes
        # D_0 = D_0.5 - 0.5 * (D_1.5 - D_0.5) = 1.5*D_0.5 - 0.5*D_1.5
        #Dpot[0]  = max(1.5 * Drawpot_nn[0]  - 0.5 * Drawpot_nn[1],  tol)
        #Dpot[-1] = max(1.5 * Drawpot_nn[-1] - 0.5 * Drawpot_nn[-2], tol)
    else:
        Dpot = np.copy(Dnode)

    # We make sure all D are always positive
    Dnode_p = positiveD(Dnode)
    Dgeo_p  = positiveD(Dgeo)
    Dpot_p  = positiveD(Dpot)

    # Smoothing 
    Dnode_s = smoother(Dnode_p)
    Dgeo_s  = smoother(Dgeo_p)
    Dpot_s  = smoother(Dpot_p)

    # D^jump, D^pot, D^geo
    # We use original D's for MFPTs and the smoothed ones for reporting
    # TODO updated: we use smoothed ones for everything
    return Dnode_p, Dnode_s, Dpot_p, Dpot_s, Dgeo_p, Dgeo_s

def is_nn(R, periodic=False, tol=0.0):
    """
    This is a function to check whether R is diffusive 
    nn=nearest_neighbor
    """
    R = np.asarray(R, float)
    n = R.shape[0]
    M = np.zeros_like(R, dtype=bool)
    i = np.arange(n)
    M[i, i] = True
    M[i[:-1], i[1:]] = True
    M[i[1:], i[:-1]] = True
    if periodic and n > 1:
        M[0, -1] = True
        M[-1, 0] = True
    return np.all(np.abs(R[~M]) <= tol)

# See: https://en.wikipedia.org/wiki/Kramers%E2%80%93Moyal_expansion
# Qi,i+1 = ui >=0, Qi,i-1 = vi >=0, Qii=-(ui+vi), Qij= 0 if |i-j| >1
# Here we replace R by a nearest-neighbor (tridiagonal) generator R' that has the 
# same 'local mean step' and 'local variance growth' per unit time at every grid point.
#
# By below, at every bin, R' agrees with R to second order
#
# Here we strictly build tridiagonal matrix with M1, M2 (and pi if we have it)
def project_nn_momentMatch(R, binwidth, periodic=False, pi=None, eps=1e-15):
    """
    This function tri-digonalize R such that the new generator matches the 1st/2nd jump moments of R;
    pi can be also passed to make sure the new R preserves detailed balance
    """
    R = np.asarray(R, float)
    n = R.shape[0]
    if n == 0:
        return R.copy(), np.array([]), np.array([])

    # Uniform grid and pairwise signed distances
    if periodic:
        dx = binwidth #width / n
        x  = np.arange(n) * dx
        #L  = width
        L = None # (I do not use this)
    else:
        #dx = width / (n - 1) if n > 1 else 1.0
        dx = binwidth
        #x  = np.linspace(0.0, width, n)
        x  = np.arange(n) * dx
        L  = None

    XX = x[None, :] - x[:, None]
    if periodic:
        XX = (XX + 0.5*L) % L - 0.5*L
    np.fill_diagonal(XX, 0.0)

    # Kramers-Moyal moments from original R
    m1 = np.sum(XX * R, axis=1)          # first moment (drift): sum delta x * R
    m2 = np.sum((XX**2) * R, axis=1)     # second moment(diffusion): sum (detla x)^2 x * R

    # Moment-matched NN rates on a uniform grid (1D):
    # We only have two rates: u_i for i to i+1; v_i for i i-1
    # m1 = dx * (u - v): avergae change per time, m2 = dx^2 * (u + v): average squared change per time
    # solve u, v
    u = 0.5 * (m2 / (dx*dx) + m1 / dx)   # i -> i+1: u = 1/2 [m2/dx^2 + m1/dx]
    v = 0.5 * (m2 / (dx*dx) - m1 / dx)   # i -> i-1: v = 1/2 [m2/dx^2 - m1/dx]

    # Ensure non-negative rates (clip tiny negatives due to noise)
    u = np.maximum(u, 0.0)
    v = np.maximum(v, 0.0)

    # Optionally enforce detailed balance with respect to pi
    if pi is not None:
        pi = np.asarray(pi, float).ravel()
        if pi.size != n:
            raise ValueError("pi must have length n = R.shape[0]")
        if np.any(pi <= 0):
            raise ValueError("pi must be strictly positive to enforce detailed balance")

        # Normalize (not necessary for DB itself, but keeps scales tidy)
        pi = pi / np.sum(pi)

        # Geometric (log-space minimal) rescaling per bond
        # For each bond (i, ip), adjust u[i] and v[ip] so that pi[i]*u[i] == pi[ip]*v[ip]
        if periodic:
            for i in range(n):
                ip = (i + 1) % n
                # skip if both are exactly zero
                if u[i] <= 0.0 and v[ip] <= 0.0:
                    continue
                alpha = np.sqrt((pi[ip] * (v[ip] + eps)) / (pi[i] * (u[i] + eps)))
                u[i] *= alpha
                v[ip] /= alpha
        else:
            # non-periodic: bonds i=0..n-2 between i and i+1
            for i in range(n - 1):
                ip = i + 1
                if u[i] <= 0.0 and v[ip] <= 0.0:
                    continue
                alpha = np.sqrt((pi[ip] * (v[ip] + eps)) / (pi[i] * (u[i] + eps)))
                u[i] *= alpha
                v[ip] /= alpha

        # Optional: clamp tiny negatives from rounding (preserves DB to ~machine eps)
        u = np.maximum(u, 0.0)
        v = np.maximum(v, 0.0)

    # Assemble tridiagonal NN generator
    Rnn = np.zeros_like(R)
    if periodic and n > 1:
        for i in range(n):
            ip = (i + 1) % n
            im = (i - 1) % n
            Rnn[i, ip] = u[i]
            Rnn[i, im] = v[i]
    else:
        if n > 1:
            Rnn[np.arange(n-1), np.arange(1, n)] = u[:-1]  # i -> i+1
            Rnn[np.arange(1, n), np.arange(n-1)] = v[1:]   # i -> i-1

    # Fix diagonals so rows sum to zero
    np.fill_diagonal(Rnn, 0.0)
    np.fill_diagonal(Rnn, -Rnn.sum(axis=1))

    return Rnn

# In the below we use project_nn_momentMatch (second momement) to fix Q
# then use "Dgeo" to estimate MFPT
def MFPT_from_Generator_M2(rateMate, fe, first=1, second=-2, delta=1,
                             binwidth=1, R=1, lag=1, kbT=1, pi=None, unit=1e-6):
    beta = 1 / kbT
    dx = binwidth
    a = rateMate
    potentialFunction = fe
    potentialFunction = potentialFunction - potentialFunction.min()

    if is_nn(a):  # if Q is already diffusive
        #print('Q is diffusive!')

        # D at the same grid as fe
        D, Ds, Dpot, Dpots, Dgeo, Dgeos = calcD(a, binswidth=dx, beta=beta, potentialFunction=potentialFunction)
        # We fix the unit of D for reporting
        Di, Dsi, Dpoti, Dpotsi, Dgeoi, Dgeosi = [x / unit for x in (D, Ds, Dpot, Dpots, Dgeo, Dgeos)]
        
        c1 = potentialFunction

        Dedge_geo = dx**2 * np.sqrt(np.diag(a, -1) * np.diag(a, 1))
        mfpt_value_fixed1 = _mfpt_from_edge_D(
            potentialFunction, Dedge_geo, first, second, binwidth, beta) * unit
        mfpt_value_fixed2 = mfpt_value_fixed1

    else:  # R is NOT already diffusive, so we force it to be

        if pi is None:
            pi = stationary_distribution_from_Generator(a)
        a_nn = project_nn_momentMatch(a, binwidth, pi=pi)
        Q = a_nn
        stat_dist1 = np.asarray(pi, dtype=float)
        stat_dist1 /= stat_dist1.sum()
        c1_t = -kbT * np.log(stat_dist1)
        c1 = c1_t - np.min(c1_t)

        # D at the same grid as fe
        D, Ds, Dpot, Dpots, Dgeo, Dgeos = calcD(Q, binswidth=dx, beta=beta, potentialFunction=potentialFunction)
        #print('Dgeo DM2 :', Dgeo)
        # We fix the unit of D for reporting
        Di, Dsi, Dpoti, Dpotsi, Dgeoi, Dgeosi = [x / unit for x in (D, Ds, Dpot, Dpots, Dgeo, Dgeos)]
        #print('Dgeo DM2 / unit :', Dgeoi)

        Dedge_geo = dx**2 * np.sqrt(np.diag(Q, -1) * np.diag(Q, 1))
        dU = np.diff(potentialFunction)
        Dedge_pot = dx**2 * np.diag(Q, -1) * np.exp(-beta * dU / 2)
        mfpt_value_fixed1 = _mfpt_from_edge_D(
            potentialFunction, Dedge_geo, first, second, binwidth, beta) * unit
        mfpt_value_fixed2 = _mfpt_from_edge_D(
            potentialFunction, Dedge_pot, first, second, binwidth, beta) * unit

    return mfpt_value_fixed1, mfpt_value_fixed2, c1, Di, Dsi, Dpoti, Dpotsi, Dgeoi, Dgeosi

# Common 1D MFPT integral using a selected diffusion profile
def _mfpt_from_D(fe, D_profile, first, second, binwidth, beta):
    direction = 1 if second > first else -1
    exp_beta_fe = np.exp(beta * fe)
    exp_minus_beta_fe = np.exp(-beta * fe)

    value = 0.0
    i = first
    while i != second:
        if i < 0 or i >= len(D_profile):
            raise IndexError("MFPT integration index is outside the D profile.")

        if direction == 1:
            inner = np.sum(exp_minus_beta_fe[first:i + 1]) * binwidth
        else:
            inner = np.sum(exp_minus_beta_fe[i:first + 1]) * binwidth

        value += exp_beta_fe[i] * inner / D_profile[i] * binwidth
        i += direction

    return value


def _mfpt_from_edge_D(fe, D_edge, first, second, binwidth, beta):
    """Discrete 1D MFPT using diffusion values on the traversed edges."""
    fe = np.asarray(fe, dtype=float)
    D_edge = np.asarray(D_edge, dtype=float)
    if len(D_edge) != len(fe) - 1:
        raise ValueError("Edge diffusion must have length len(fe)-1.")
    if np.any(~np.isfinite(D_edge)) or np.any(D_edge <= 0):
        return float('nan')

    n_ = np.exp(-beta * (fe - np.min(fe)))
    value = 0.0
    if first < second:
        for i in range(first, second):
            inner = np.sum(n_[first:i + 1])
            n_edge = np.sqrt(n_[i] * n_[i + 1])
            value += inner / (D_edge[i] * n_edge)
    else:
        for i in range(first, second, -1):
            inner = np.sum(n_[i:first + 1])
            n_edge = np.sqrt(n_[i] * n_[i - 1])
            value += inner / (D_edge[i - 1] * n_edge)
    return value * binwidth**2


# MFPT via D_geo
def MFPT_from_Generator_Dgeo(rateMat, fe, first=1, second=-2, delta=1,
                              binwidth=1, R=1, lag=1, kbT=1, unit=1e-3):
    beta = 1 / kbT
    D, Ds, Dpot, Dpots, Dgeo, Dgeos = calcD(
        rateMat, binswidth=binwidth, beta=beta, potentialFunction=fe)
    reported = [x / unit for x in (D, Ds, Dpot, Dpots, Dgeo, Dgeos)]
    Dedge = binwidth**2 * np.sqrt(np.diag(rateMat, -1) * np.diag(rateMat, 1))
    mfpt = _mfpt_from_edge_D(fe, Dedge, first, second, binwidth, beta) * unit
    return mfpt, *reported


# MFPT via D_pot
def MFPT_from_Generator_Dpot(rateMat, fe, first=1, second=-2, delta=1,
                              binwidth=1, R=1, lag=1, kbT=1, unit=1e-3):
    beta = 1 / kbT
    D, Ds, Dpot, Dpots, Dgeo, Dgeos = calcD(
        rateMat, binswidth=binwidth, beta=beta, potentialFunction=fe)
    reported = [x / unit for x in (D, Ds, Dpot, Dpots, Dgeo, Dgeos)]
    dU = np.diff(fe)
    Dedge = binwidth**2 * np.diag(rateMat, -1) * np.exp(-beta * dU / 2)
    mfpt = _mfpt_from_edge_D(fe, Dedge, first, second, binwidth, beta) * unit
    return mfpt, *reported


# MFPT via D_jump
def MFPT_from_Generator_Djump(rateMat, fe, first=1, second=-2, delta=1,
                               binwidth=1, R=1, lag=1, kbT=1, unit=1e-3):
    beta = 1 / kbT
    D, Ds, Dpot, Dpots, Dgeo, Dgeos = calcD(
        rateMat, binswidth=binwidth, beta=beta, potentialFunction=fe)
    reported = [x / unit for x in (D, Ds, Dpot, Dpots, Dgeo, Dgeos)]
    mfpt = _mfpt_from_D(fe, Ds, first, second, binwidth, beta) * unit
    return mfpt, *reported

def MFPT_from_Generator_Poisson(Q, first=1, second=20, binwidth=1, delta=1, R=1, lag=1, unit=1e-6):
    """
    MFPT by solving -Q_ss * t_s = 1
    Returns a scalar MFPT in unit, or nan if the system cannot be solved.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    i = int(first)
    j = int(second)
    # auto-detect 1-based indexing (e.g., first=1, second=n)
    if i >= n or j >= n:
        i -= 1
        j -= 1
    if not (0 <= i < n and 0 <= j < n):
        warnings.warn("State indices out of range after index adjustment.")
        return float('nan')
    # S = all states except the target j
    S_mask = np.ones(n, dtype=bool)
    S_mask[j] = False
    S = np.where(S_mask)[0]
    # Solve: -Q_SS * t_S = 1, with t_j = 0
    A = -Q[np.ix_(S, S)]
    b = np.ones(len(S))
    try:
        t_S = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        warnings.warn("Subgenerator is singular; target may be unreachable from some states.")
        return float('nan')
    # assemble full t vector
    t = np.zeros(n)
    t[S] = t_S
    t[j] = 0.0
    return t[i] * unit

def timescales_from_Q(Q, k=3):
    """
    Returns the top k implied timescales from Rate Matrix Q.
    """
    # 1. Calculate Eigenvalues
    evals = np.linalg.eigvals(Q)

    # 2. Sort them.
    # For a Rate Matrix, we want values closest to 0 from the negative side.
    # Usually they are 0, -0.001, -0.05 ...
    # precise sorting: Real parts, descending
    evals = np.sort(np.real(evals))[::-1]

    # 3. Calculate Timescales t = -1 / lambda
    timescales = []
    # Skip index 0 (which is the stationary state, lambda=0)
    for i in range(1, k + 1):
        if i < len(evals):
            val = evals[i]
            # Safety: only valid if eigenvalue is negative
            if val < 0:
                t = -1.0 / val
                timescales.append(t)
            else:
                timescales.append(np.nan) # Positive rate = unphysical or noise
        else:
            timescales.append(np.nan)

    return timescales

##################
# FE estimators
##################
def stationary_distribution_from_Generator(Q):
    """
    Returns the stationary distribution mu for the continuous-time
    rate matrix Q (shape=(n,n)). We solve:
        mu Q = 0   and   sum(mu) = 1
    """

    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]

    # 1) Build the augmented system A x = b,
    #    where A has (n-1) rows from Q^T, plus one row enforcing sum(mu)=1.
    A = np.zeros((n, n))
    b = np.zeros(n)

    # Copy first (n-1) rows from R^T
    # We want mu Q = 0 => Q^T mu^T = 0 => (Q^T)[:(n-1), :] * mu^T = 0
    A[:(n-1), :] = Q.T[:(n-1), :]

    # Last row => sum(mu) = 1
    A[n-1, :] = 1.0
    b[n-1] = 1.0

    # 2) Solve A mu = b
    mu, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)

    if rank < n:
        raise ValueError("Generator does not have a unique stationary distribution.")
    if np.any(mu < -1e-12):
        raise ValueError("Generator has a negative stationary probability.")

    # 3) Remove only tiny negative numerical noise
    mu = np.maximum(mu, 0.0)

    # 4) Renormalize and check the stationary residual
    if mu.sum() <= 0 or np.any(mu == 0):
        raise ValueError("Generator stationary distribution is not strictly positive.")
    mu = mu / np.sum(mu)
    scale = max(1.0, np.max(np.abs(Q)))
    if np.max(np.abs(mu @ Q)) > 1e-8 * scale:
        raise ValueError("Generator stationary residual is too large.")
    return mu


def stationary_distribution_from_Generator_eig(Q):
    """
    Find stationary distribution from the left eigenvector
    of Q for eigenvalue 0.

    We can do that by looking at Q^T's right eigenvectors for eigenvalue 0.
    """
    # 1) Eigen-decomposition of Q^T
    w, v = np.linalg.eig(Q.T)
    # w: eigenvalues, v: columns are eigenvectors

    # 2) Find index of eigenvalue closest to 0 (smallest EV)
    idx_zero = np.argmin(np.abs(w))

    # 3) Right eigenvector 'v[:, idx_zero]' => corresponds to left eigenvector of Q
    mu = np.real(v[:, idx_zero])  # just in case it has small imag part

    # 4) Ensure nonnegative
    mu = np.abs(mu)

    # 5) Normalize to sum=1
    mu /= mu.sum()
    return mu

def detailedBalance(mat, ij=True):
    FE = np.zeros(mat.shape[0])
    for i in range(mat.shape[0]-1):
        if(mat[i+1,i]==0 or mat[i,i+1]==0):
            FE[i] = 0
        else:
            if ij:
                FE[i] = FE[i-1] - kbT * np.log(mat[i,i+1]/mat[i+1,i])
            else:
                FE[i] = FE[i-1] - kbT * np.log(mat[i+1,i]/mat[i,i+1])
    FE = FE - FE.min()
    return FE

def stationary_distribution_from_FE(F, kbT=1.0):
    F -= np.min(F)
    weights = np.exp(-F / kbT)
    pi = weights / np.sum(weights)
    return pi

#######################################
## for tests purposes
#######################################
def timescales_from_CountMatrix(C, lag=1.0, k=3):
    """
    Returns the top k implied timescales from an empirical Count Matrix C.
    """
    C = np.asarray(C, dtype=float)

    # 1. Normalize count matrix to get Transition Matrix P
    row_sums = np.sum(C, axis=1)

    # Avoid division by zero if a state has no outgoing counts
    with np.errstate(divide='ignore', invalid='ignore'):
        P = C / row_sums[:, np.newaxis]

    # If a row summed to zero, the division resulted in NaNs.
    # Set those rows to zero (absorbing state) or a uniform distribution.
    P[np.isnan(P)] = 0.0

    # 2. Calculate Eigenvalues
    evals = np.linalg.eigvals(P)

    # 3. Sort them.
    # For a Transition Matrix, we want values closest to 1.
    # Usually they are 1.0, 0.99, 0.85 ...
    # precise sorting: Real parts, descending
    evals = np.sort(np.real(evals))[::-1]

    # 4. Calculate Timescales t = -lag / ln(lambda)
    timescales = []
    # Skip index 0 (which is the stationary state, lambda=1)
    for i in range(1, k + 1):
        if i < len(evals):
            val = evals[i]
            # Safety: only valid if eigenvalue is strictly between 0 and 1
            if 0 < val < 1.0:
                t = -lag / np.log(val)
                timescales.append(t)
            else:
                timescales.append(np.nan) # Negative, zero, or >= 1 = unphysical or noise
        else:
            timescales.append(np.nan)

    return timescales

def D_from_P(P, tau, binwidth):
    """Local diffusion coefficient directly from transition matrix P.
    No matrix log, no embeddability problem."""
    K = P.shape[0]
    offset = np.arange(K)[None, :] - np.arange(K)[:, None]   # (j - i), shape (K,K)
    D_i = (binwidth**2 / (2 * tau)) * np.sum(P * offset**2, axis=1)
    return D_i

