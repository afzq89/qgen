#!/usr/bin/python3

import numpy as np
#from qpsolvers import solve_qp
#from scipy import optimize 


# The logarithm is divided by the (physical) lag time to obtain a generator, so a loose
# absolute tolerance can become a large rate error at short lags.
eps = 1e-14 # In estimating log(M), I devide log(M) by tau after the calculations in isRealLog, 
            # isRealLogTaylor, isRealLoglogm, so eps should be large indeed ~ 1e-3
            # but _safe_logm, I have already devided by tau!
maxIters = 10000000

###########################################################
## General helpers for normalization & log
###########################################################
#Function to Normalize a square Matrix.
#k=0 is row normalization, k=1 is column normalization.
#Row Normalization corresponds to i->j
#Column Normalization corresponds to j->i
#FIXME: for now I skip the matrices that are not normalizable; this
#       means we did not sample enough;
#       we should find a way for the cases that we did not sample
def normalizeMatrix(matrix, k=0):
    normalizedMatrix = np.copy(matrix).astype(float)
    with np.errstate(divide='raise', invalid='raise'):
        try:
            if k == 0:  # Row normalization
                for i in range(normalizedMatrix.shape[0]):
                    normalizedMatrix[i] = normalizedMatrix[i] / normalizedMatrix[i].sum()
            else:  # Column normalization
                for i in range(normalizedMatrix.shape[1]):
                    normalizedMatrix[:, i] = normalizedMatrix[:, i] / normalizedMatrix[:, i].sum()
        except FloatingPointError as e:
            raise ValueError("Matrix is not normalizable: encountered division by zero or invalid value.") from e
    return normalizedMatrix

# Vectorize normalization
def _row_normalize(P):
    P = np.array(P, dtype=float)
    row_sums = P.sum(axis=1, keepdims=True)
    if np.any(row_sums == 0):
        raise ValueError("Matrix has a zero row and cannot be normalized.")
    return P / row_sums

#frobenius is a measure of distance between two matrices
#this function takes two square matrices (q1, q2) and finds their frobenius distance
#find the distance between the two matrices
def frobenius(q1, q2):
#Arguments:#
#q1/2 - input matrices to calculate distance

    n = q1.shape[0]
    dist = 0.0
    for i in range(n):
        for j in range(n):
            dist += (abs(q1[i,j]) - abs(q2[i,j]))**2
    return np.sqrt(dist)

#Iteratively take the log of a matrix.  This is a function which always produces real numbers
# log(A) =  sum_n (-1)^n+1/n (A-I)^n
def isRealLog(matrix, epsilon=eps, maxIterations=maxIters):
#Arguments:#
#matrix - take the log of this matrix
#epsilon - convergenace criterion
#maxIterations - maximum iterations until we just say it has converged arbitrarily
    # it is already normalized
    #matrix = normalizeMatrix(matrix)
    iterations = 1
    ret = np.zeros((matrix.shape))
    previousIteration = np.copy(matrix)
    i=1
    ret = ret + float(i)/float(iterations) * (matrix - np.identity(len(matrix[0])))**iterations
    distance = frobenius(previousIteration, ret)
    while(distance>epsilon):
        iterations+=1
        i=i*-1
        ret = ret + float(i)/float(iterations) * (matrix - np.identity(len(matrix[0])))**iterations
        distance = frobenius(ret, previousIteration)
        previousIteration = np.copy(ret)
        if iterations == maxIterations+1:
            distance = 0
    #print("Distace: %s" %distance)
    #print("Log Iterations: %s" %iterations)
    return ret


# Iteratively calculate a real matrix logarithm with the Taylor series
# log(M) = log(I+delta), delta = M-I
# log(M) = sum_n (-1)^(n+1) (M-I)^n / n.
# The series converges only when rho(M-I) <= 1.
def isRealLogTaylor(matrix, epsilon=eps, maxIterations=maxIters):
    """Return ``log(matrix)`` using matrix powers.
    """
    matrix = np.asarray(matrix, dtype=float)

    delta = matrix - np.eye(matrix.shape[0]) # delta = M - I
    spectral_radius = np.max(np.abs(np.linalg.eigvals(delta))) # max eigenValue of delta
    if not np.isfinite(spectral_radius) or spectral_radius > 1.0:
        raise ValueError(
            "Matrix-log Taylor series requires spectral radius "
            f"rho(P-I) <= 1; got {spectral_radius:.6g}.")

    # Updated
    power = delta.copy()
    result = power.copy()
    # if M is close to I, return it
    if np.linalg.norm(power, ord="fro") <= epsilon:
        return result
    # otherwise, use the below Taylor expansion
    for iteration in range(2, maxIterations + 1):
        power = power @ delta
        coefficient = 1.0 / iteration if iteration % 2 else -1.0 / iteration
        increment = coefficient * power
        result += increment
        if np.linalg.norm(increment, ord="fro") <= epsilon:
            #print("Log Iteration: %s" %iteration)
            #print("Distance: ", np.linalg.norm(increment, ord="fro"))
            return result

    raise RuntimeError(
        "Matrix-log Taylor series did not converge within "
        f"{maxIterations} iterations (epsilon={epsilon:g}).")

# Log matrix via scipy logm (this is redudant to _safe_logm)
from scipy.linalg import logm, expm
def isRealLogLogm(matrix, epsilon=eps, maxIterations=maxIters):
    """Return log(matrix) via logm scipy.linalg
    """
    matrix = np.asarray(matrix, dtype=float)

    result = logm(matrix)

    if np.max(np.abs(result.imag)) > epsilon:
        raise ValueError("Matrix logarithm is not real.")

    error = np.linalg.norm(expm(result.real) - matrix, ord="fro") / np.linalg.norm(matrix, ord="fro")
    if error > epsilon:
        raise ValueError(f"Matrix logarithm reconstruction error is too large: {error:.3e}")

    return result.real

# Safe log using logm of scipy with symmetrization feature
def _safe_logm(P, tau, symmetrize=True, epsilon=1.0):
    if symmetrize:
        # Use P^2 = P(2*tau) to suppress negative eigenvalues before logm; 
        # and divide by 2*tau: P^2 = P(tau)P(tau)~P(2tau) 
        P2 = P @ P # P @ P instead of P @ P.T
        L = np.real(logm(P2)) / (2 * tau)

        error = np.linalg.norm(expm(L*2*tau) - P2, ord="fro") / np.linalg.norm(P2, ord="fro")
        if error > epsilon:
            raise ValueError(f"Matrix logarithm reconstruction error is too large: {error:.3e}")
    else:
        evals = np.linalg.eigvals(P)
        neg = [e for e in evals if np.isreal(e) and np.real(e) < 0]
        if neg:
            print(f"Warning: P has {len(neg)} negative real eigenvalue(s). "
                  f"Imaginary parts of logm will be discarded.")
        L = np.real(logm(P)) / tau

        error = np.linalg.norm(expm(L*tau) - P, ord="fro") / np.linalg.norm(P, ord="fro")
        if error > epsilon:
            raise ValueError(f"Matrix logarithm reconstruction error is too large: {error:.3e}")
    return L


#Simply take a NxN numpy array and write it to a file.
def writeMatrix(matrix, fileName):
    file = open(fileName, "w")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            file.write("%.6e\t"%matrix[i,j])
        file.write("\n")
    file.close()

###########################################################
## Estimators start here
###########################################################

#######################################
## DA
#######################################
def diagonalAdjustment(matrix, tau=1, k=0, epsilon=eps, maxIterations=maxIters, LogMethod='pseudo'):
    try:
        #matrix = normalizeMatrix(matrix, k=k)
        matrix = _row_normalize(matrix)
    except ValueError as e:
        raise ValueError("diagonalAdjustment: normalization failed, cannot proceed: " + str(e))
    if LogMethod == 'safetaylor':
       logMatrix = isRealLogTaylor(matrix, epsilon=epsilon, maxIterations=maxIterations) / tau
    elif LogMethod == 'taylor':
       logMatrix = isRealLog(matrix, epsilon=epsilon, maxIterations=maxIterations) / tau
    elif LogMethod == 'safelog':
         logMatrix = _safe_logm(matrix, tau, symmetrize=False, epsilon=1.0)
    elif LogMethod == 'pseudo':
         I = np.eye(matrix.shape[0])
         logMatrix = (matrix - I) / tau 

    for i in range(logMatrix.shape[0]):
        for j in range(logMatrix.shape[0]):
            if(i!=j and logMatrix[i,j]<0):
                logMatrix[i,j] = 0

    #make diagonals the negative sum of rest of row
    for i in range(logMatrix.shape[0]):
        logMatrix[i,i]=0 #first set diagonals to zero
        logMatrix[i,i] = -1 * logMatrix[i].sum() #by row
    return logMatrix

#######################################
## MLE with fixed FE
#######################################
from scipy.linalg import eigh
from scipy.optimize import minimize

def build_Q(D, F, dx, kbT):
    """Tridiagonal rate matrix"""
    K = len(F)
    i = np.arange(K - 1)
    dF = np.diff(F)
    Q = np.zeros((K, K))
    Q[i, i + 1] = (D / dx**2) * np.exp(-dF / (2 * kbT))
    Q[i + 1, i] = (D / dx**2) * np.exp(+dF / (2 * kbT))
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def _eig(D, F, dx, kbT):
    """Eigendecomposition of S = Pi^(1/2) Q Pi^(-1/2).

    S is symmetric (detailed balance) with off-diagonal D/dx^2, so this
    replaces expm.
    """
    K = len(F)
    i = np.arange(K - 1)
    Q = build_Q(D, F, dx, kbT)
    S = np.zeros((K, K))
    S[i, i + 1] = S[i + 1, i] = D / dx**2
    np.fill_diagonal(S, np.diag(Q))
    w, U = eigh(S)
    return Q, w, U

# L = -\sum C_ij log P_ij + lambda* R
def _nll_grad(logD, F, C, dx, kbT, tau, lam):
    """Negative log-likelihood and its exact gradient w.r.t. log D."""
    D = np.exp(logD)
    K = len(F)
    i = np.arange(K - 1)
    Q, w, U = _eig(D, F, dx, kbT)
    a = w * tau
    ratio = np.exp((F[:, None] - F[None, :]) / (2 * kbT))
    P = np.maximum(((U * np.exp(a)) @ U.T) * ratio, 1e-30)

    m = C > 0
    nll = -np.sum(C[m] * np.log(P[m]))

    B = np.zeros_like(P)
    B[m] = -C[m] / P[m]
    B *= ratio

    # derivative of a symmetric matrix exponential (Daleckii-Krein):
    # W[m,n] = (e^a_m - e^a_n) / (a_m - a_n), written to avoid overflow
    dd = np.abs(a[:, None] - a[None, :])
    small = dd < 1e-8
    safe = np.where(small, 1.0, dd)
    W = np.exp(np.maximum(a[:, None], a[None, :])) * np.where(
        small, 1.0 - 0.5 * dd, -np.expm1(-safe) / safe)
    R = U @ (U.T @ B @ U * W).T @ U.T

    grad = tau * (-R[i, i] * Q[i, i + 1] - R[i + 1, i + 1] * Q[i + 1, i]
                  + (R[i, i + 1] + R[i + 1, i]) * D / dx**2)

    if lam > 0:
        L = np.diff(np.eye(K - 1), n=2, axis=0)
        nll += lam * np.sum(np.diff(logD, n=2) ** 2)
        grad += 2 * lam * (L.T @ L @ logD)
    return nll, grad


def _initial_D(C, F, dx, kbT, tau):
    """Short-lag estimate: k(i->i+1) ~ C[i,i+1] / (N_i * tau)."""
    N = np.maximum(C.sum(axis=1), 1.0)
    i = np.arange(len(F) - 1)
    dF = np.diff(F)
    D = 0.5 * dx**2 / tau * (C[i, i + 1] / N[i] * np.exp(+dF / (2 * kbT))
                             + C[i + 1, i] / N[i + 1] * np.exp(-dF / (2 * kbT)))
    return np.where(D > 0, D, np.median(D[D > 0]) if np.any(D > 0) else dx**2 / tau)


def _edge_initial_D(D0, K):
    D0 = np.asarray(D0, float)
    if D0.shape == (K - 1,):
        return D0
    if D0.shape == (K,):
        return 0.5 * (D0[:-1] + D0[1:])
    raise ValueError(f"D0 must have length {K - 1} or {K}; got shape {D0.shape}.")


def MLE(C, F, dx, kbT, tau, lam=0.0, D0=None, disp=False):
    """Fit D(x) on the K-1 bin edges from a K x K RAW count matrix."""
    C = np.asarray(C, float)
    F = np.asarray(F, float)

    # Some checks
    i = np.arange(len(F) - 1)
    near = C[i, i + 1].sum() + C[i + 1, i].sum()
    if near == 0:
        raise ValueError(
            "Count matrix has no transitions between adjacent bins, so there "
            "is nothing to fit. Usual cause")
    if near < 0.001 * C.sum():
        print(f"Warning: only {100 * near / C.sum():.3f}% of counts are moves "
              "to an adjacent bin. The lag is probably far too short/long.")

    # initial D guess
    x0 = np.log(_initial_D(C, F, dx, kbT, tau) if D0 is None else _edge_initial_D(D0, len(F)))
    # MULTIPLE initial D
    from scipy.ndimage import gaussian_filter1d
    x_const  = np.full_like(x0, np.median(x0))
    x_smooth = gaussian_filter1d(x0, sigma=1.0)
    starts = [x0, x0 + np.log(0.1), x0 + np.log(10.0), x_const, x_smooth,]
    # bounds
    bounds = [ (v - 10.0, v + 10.0) for v in x0 ]
    ## TODO ##
    Copt = C / C.sum()
    args = (F, Copt, dx, kbT, tau, lam)
    # options
    opts = {"ftol": 1e-12, "gtol": 1e-7, "maxiter": 10000, "maxfun": 200000, "maxls": 100,
            "maxcor": 20, "disp": disp}

    # Optimize
    results = []
    #for s in (np.log(0.1), 0.0, np.log(10.0)):
    for start in starts:
    	# The starting guess can be off by a large constant factor, which strands
   	# L-BFGS-B. Retry from a few global rescalings and keep the best.
        res = minimize(_nll_grad, start, args=args, jac=True, method="L-BFGS-B",
                       bounds=bounds, options=opts)
        if res.success and np.isfinite(res.fun) and np.all(np.isfinite(res.jac)):
            results.append(res)
    if not results:
       raise RuntimeError("All MLE optimizations failed.")

    best = min(results, key=lambda r: r.fun)

    start_fun = _nll_grad(x0, *args)[0]
    if best.fun > start_fun + 1e-10 * max(1.0, abs(start_fun)):
        #raise RuntimeError("Optimiser ended worse than its starting guess.")
         print("Warning: optimiser ended worse than its starting guess. Do not use.")

    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    if np.any(best.x <= lower + 1e-7) or np.any(best.x >= upper - 1e-7):
        #raise RuntimeError("Optimiser ended at a diffusion bound.")
        print("Warning: optimiser ended worse than its starting guess. Do not use.")

    D = np.exp(best.x)
    r = (D * tau / dx**2).max()
    if r > 1.5:
        #raise RuntimeError(
        print(
            f"Warning: D*tau/dx^2 reaches {r:.1f}. Past ~1 the likelihood stops "
            f"distinguishing individual edges. Use a shorter lag than {tau}.")

    Dnode = np.zeros(C.shape[0])
    Dnode[1:-1] = 0.5 * (D[:-1] + D[1:])
    Dnode[0] = D[0]
    Dnode[-1] = D[-1]

    return Dnode, build_Q(D, F, dx, kbT)

#######################################
## MLE for biased MD with fixed FE
#######################################
def MLE_biased(C_list, F_list, dx, kbT, tau, FE_unbiased, lam=0.0, D0=None, disp=False):
    """Fit ONE D(x) shared across umbrella windows.

    C_list[w] : K x K RAW count matrix from window w. We do not reweight it.
    F_list[w] : BIASED free energy of window w at the bin centres, i.e.
                F + bias_w. This is comming from window's own
                populations, F_list[w] = -kbT * log(p_w), which needs no
                knowledge of the restraint form.
    FE_unbiased : unbiased FE to be used for the final reported Q
    """
    C_list = [np.asarray(C, float) for C in C_list]
    F_list = [np.asarray(F, float) for F in F_list]

    if D0 is None:
        est = np.array([_initial_D(C, F, dx, kbT, tau)
                        for C, F in zip(C_list, F_list)])
        x0 = np.log(np.median(est, axis=0))
    else:
        x0 = np.log(np.asarray(D0, float))

    def obj(logD):
        nll, grad = 0.0, np.zeros_like(logD)
        for C, F in zip(C_list, F_list):
            n, g = _nll_grad(logD, F, C, dx, kbT, tau, 0.0)
            nll += n
            grad += g
        if lam > 0:
            L = np.diff(np.eye(len(logD)), n=2, axis=0)
            nll += lam * np.sum(np.diff(logD, n=2) ** 2)
            grad += 2 * lam * (L.T @ L @ logD)
        return nll, grad

    opts = {"ftol": 1e-14, "gtol": 1e-10, "maxiter": 5000, "maxfun": 50000,
            "disp": disp}
    best = None
    for s in (np.log(0.1), 0.0, np.log(10.0)):
        res = minimize(obj, x0 + s, jac=True, method="L-BFGS-B",
                       bounds=[(v + s - 8, v + s + 8) for v in x0], options=opts)
        if best is None or res.fun < best.fun:
            best = res
    if best.fun > obj(x0)[0]:
        print("Warning: optimiser ended worse than its starting guess.")

    D = np.exp(best.x)
    r = (D * tau / dx**2).max()
    if r > 1.5:
        print(f"Warning: D*tau/dx^2 reaches {r:.1f}. Past ~1 the likelihood "
              "stops distinguishing individual edges -- D can be far off with a "
              f"perfectly good likelihood. Use a shorter lag than {tau}.")

    Dnode = np.zeros(len(FE_unbiased))
    Dnode[1:-1] = 0.5 * (D[:-1] + D[1:])
    Dnode[0] = D[0]
    Dnode[-1] = D[-1]

    return Dnode, build_Q(D, FE_unbiased, dx, kbT)

#######################################
## PolySolver by Yuchen Xu JPCA 2023
#######################################
from qpsolvers import solve_qp
def polySolver(matrix, tau=1, k=0, epsilon = eps, maxIterations=maxIters, thresh = 1.e-8):
    matrix = normalizeMatrix(matrix, k=k)
    logMatrix = isRealLog(matrix, epsilon=epsilon, maxIterations=maxIterations) / tau
    d = len(matrix)

    # Create a sparse matrix SpMat where each column is a flattened version
    # of a d by d matrix (i.e a vector) that is obtained by iteratively multiplying an initial d by d
    # identity (temp) by matrix: equation before 12
    SpMat = np.zeros((d*d, d))
    temp = np.identity(d)
    for i in range(d):
        SpMat[:,i] = temp.flatten('F')
        temp = np.matmul(temp, matrix)

    # SVD for the independent basis
    u,s,_ = np.linalg.svd(SpMat, full_matrices=False)
    # Apply a thresholding operation to s to remove small singular values and corresponding columns of u:
    # selects the columns of u and the corresponding singular values in s where the absolute value of the
    # singular values is greater than a certain threshold thresh.
    SpMat, D = u[:,(abs(s) > thresh)]*s[abs(s) > thresh], sum(abs(s) > thresh)

    # P, q are the main the rest are constrains?
    # P,q,G,h,A,b follow the notations of function solve_qp
    P = np.matmul(SpMat.T, SpMat)
    q = - np.matmul(SpMat.T, logMatrix.flatten('F')).reshape(D,)
    e1 = np.zeros(d,)
    e1[0] = 1
    A = np.matmul(np.kron(np.ones(d), e1), SpMat)
    b = np.zeros((1,))
    G = - np.matmul(np.identity(d**2)[np.identity(d).flatten() != 1,:], SpMat)
    h = np.zeros((len(G),))
    xhat = solve_qp(P,q,G,h,A,b,verbose=True) #,eps_rel = 1.0e-06)

    mat = np.matmul(SpMat, xhat).reshape((d,d), order='F')
    # In some cases the returned matrix may have some numerically negative off-diagonal values,
    # but it is acceptable to set those negligible terms to zero manually.
    for i in range(d):
        for j in range(d):
            if i != j and mat[i,j] < 0:
                mat[i, j] = 0

    return mat

#######################################
## CWO
#######################################
# Helpers for CWO
#iteratively take the exponential of a matrix as per the expansion above.
def iterative_expm(matrix, epsilon=eps, maxIterations=maxIters):
    """Calculate a matrix exponential with its Taylor series."""
    matrix = np.asarray(matrix, dtype=float)

    result = np.eye(matrix.shape[0])
    term = np.eye(matrix.shape[0])
    for iteration in range(1, maxIterations + 1):
        term = (term @ matrix) / iteration
        result += term
        if np.linalg.norm(term, ord="fro") <= epsilon:
            return result

    raise RuntimeError(
        "Matrix-exponential Taylor series did not converge within "
        f"{maxIterations} iterations (epsilon={epsilon:g})."
    )

#Helper function by which to optimize the frobenius distance between the two matrices.
def optimizeFunc(x, i, j, Q, P):#x is exp(Q(q_{i,j}))
    Q_local = Q.copy()
    Q_local[i, i] += Q_local[i, j] - x
    Q_local[i, j] = x
    return frobenius(iterative_expm(Q_local), P)

def CWO(matrix, tau=1, k=0, epsilon=eps, maxIterations=maxIters):
    calculations = 0
    # It is noted that any method can be used here, not just diagonal adjustment (DA).
    Q = diagonalAdjustment(matrix, tau=tau, k=k, epsilon=epsilon, maxIterations=maxIterations)
    try:
        # Normalize the adjusted matrix and store the result in Q.
        matrix = normalizeMatrix(matrix, k=k)
    except ValueError as e:
        raise ValueError("CWO: normalization failed, cannot proceed: " + str(e))

    # Loop over all off-diagonal entries in Q and optimize them if greater than the threshold.
    for i in range(Q.shape[0]):
        for j in range(Q.shape[0]):
            if i != j:
                if Q[i, j] > 1e-10:
                    calculations += 1
                    # Run an optimization on the given entry with Q[i,j] as the initial guess.
                    optimalVal = optimize.fmin(optimizeFunc, Q[i, j], args=(i, j, Q, matrix),
                                                maxiter=200, full_output=False, disp=False)
                    # If fmin returns an iterable (e.g., an array), take the first element.
                    Q[i, j] = optimalVal[0] if hasattr(optimalVal, '__iter__') else optimalVal
    return Q

#######################################
## QOG
#######################################
# Helpers for QOG
def findMValue(array): #step 3 of algorithm
    n = len(array)-1 #last index
    val=0
    for i in range(1,n+1): #i loops from 1 to n
        val = (n+1-i)*array[i+1]-array[0]
        for j in range(n-i):#from 0 to n-1-i
            val -= array[n-j]
        if(val>=0): #truth condition of algorithm
            return i

    return -1 #otherwise return that row cannot be optimized.

def sortMatrix(matrix): #returns sortMatrix and unsortKey
    sortMatrix = np.copy(matrix)
    for i in range(matrix.shape[0]):
        sortMatrix[i].sort()
    sortMatrix = sortMatrix

    unsortKey = np.zeros((matrix.shape[0], matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            f=0
            while(unsortKey[i,j]==0):
                if(sortMatrix[i,f]==matrix[i,j]):
                    unsortKey[i,j] = f + 1
                f+=1
    return sortMatrix, unsortKey

def unsortMatrix(matrix, key): #take in sorted matrix and key to unsort
    unsortedMatrix = np.zeros((matrix.shape[0],matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[0]):
            unsortedMatrix[i,j] = matrix[i,int(key[i,j])-1]
    return unsortedMatrix

def QOG(matrix, tau=1, k=0, epsilon=eps, maxIterations=maxIters):
    try:
        matrix = normalizeMatrix(matrix, k=k)
    except ValueError as e:
        raise ValueError("QOG: normalization failed, cannot proceed: " + str(e))
    logMatrix = isRealLog(matrix, epsilon=epsilon, maxIterations=maxIterations) / tau

    #step 2 of algorithm
    sortedMatrix, unsortKey = sortMatrix(logMatrix)

    #step 3 of algorithm
    m = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        m[i] = findMValue(sortedMatrix[i])

    #step 4 of algorithm
    copyMatrix=np.copy(sortedMatrix)
    for i in range(matrix.shape[0]):#for each row
        for j in range(2,int(m[i])+1):#include m[i]
            sortedMatrix[i,j]=0
        for j in range(int(m[i])+1,matrix.shape[0]):#for each value not zero'd
            for k in range(int(m[i])+1,matrix.shape[0]): #summation
                sortedMatrix[i,j] -= copyMatrix[i,k] / (matrix.shape[0] - m[i] + 1)
            sortedMatrix[i,j] -= copyMatrix[i,0] / (matrix.shape[0] - m[i] + 1)
        for k in range(int(m[i]+1),matrix.shape[0]):
            sortedMatrix[i,0] -= copyMatrix[i,k] / (matrix.shape[0] - m[i] + 1)
        sortedMatrix[i,0] -= copyMatrix[i,0] / (matrix.shape[0] - m[i] + 1)

    #step 5 - shuffle rows back into order.
    quasi = unsortMatrix(sortedMatrix, unsortKey)
    return quasi

#######################################
## QP estimator adapted by Yuchen Xu
# JPCA 2023
#######################################
def qp_rate(matrix, tau):
    matrix = normalizeMatrix(matrix, 0)

    # Eigen-decomposition
    D, Phi = LA.eig(matrix)
    Psi = LA.inv(Phi)
    logLam = np.log(D)

    n = len(matrix)

    # Define the selection matrix for off-diagonal elements
    S = np.zeros((n**2, n*(n-1)))
    for i in range(n):
        ind = [j for j in range(n) if j != i]
        temp = np.zeros((n, n-1))
        temp[i,:] = -1
        temp[ind,:] = np.eye(n-1)
        vec = np.zeros(n); vec[i] = 1
        S += np.kron(temp, np.diag(vec))

    # P,q,lb follow the notations of function solve_qp.
    # Weights: alpha_k = |psi_k|^2, beta_k = |phi_k|^2, gamma_k = |psi_k|^2 * |phi_k|^2.
    # Only the off-diagonal entries are set to be variables.
    P = np.zeros((n**2,n**2))
    q = np.zeros((1,n**2))
    for i in range(n):
        P += np.kron(np.eye(n), Psi[i:i+1,:].conj().T @ Psi[i:i+1,:])/(LA.norm(Psi[i,:]))**2
        P += np.kron(Phi[:,i:i+1].conj() @ Phi[:,i:i+1].T, np.eye(n))/(LA.norm(Phi[:,i]))**2
        P += np.kron(Phi[:,i:i+1].conj() @ Phi[:,i:i+1].T, Psi[i:i+1,:].conj().T @ Psi[i:i+1,:])/((LA.norm(Psi[i,:]))**2 * (LA.norm(Phi[:,i]))**2)#/(abs(logLam[i]))**2

        q += logLam[i]*np.kron(Psi[i:i+1,:], Psi[i:i+1,:].conj())/(LA.norm(Psi[i,:]))**2
        q += logLam[i]*np.kron(Phi[:,i:i+1].conj().T, Phi[:,i:i+1].T)/(LA.norm(Phi[:,i]))**2
        q += logLam[i]*np.kron(Phi[:,i:i+1].conj().T, Psi[i:i+1,:].conj())/((LA.norm(Psi[i,:]))**2 * (LA.norm(Phi[:,i]))**2)#/(abs(logLam[i]))**2
    q += q.conj()
    q = - (q @ S).reshape(n*(n-1),)
    P = np.matmul(S.T, np.matmul(P, S))
    P = P + P.T
    lb = np.zeros((n*(n-1),))
    #xx = solve_qp(P = P, q = q, lb = lb).reshape((n,n-1), order = 'F')
    xx = solve_qp(P=P, q=q, lb=lb, solver="quadprog").reshape((n, n-1), order = 'F')

    # reproduce the rate matrix from the off-diagonal entries
    mat = np.zeros((n, n))
    for i in range(n):
        ind = [j for j in range(n) if j != i]
        mat[i,ind] = xx[i,:]
        mat[i,i] = - np.sum(xx[i,:])
    mat = mat/tau
    for i in range(n):
        for j in range(n):
            if i != j and mat[i,j] < 0:
                mat[i,j] = 0
    return mat

#######################################
## polySolver2
#######################################
def polySolver2(P, tau=1.0, epsilon=eps, maxIterations=maxIters, symmetrize=True, thresh=1e-9, LogMethod='diagonaladjustment'):
    """
    Estimates a valid rate matrix Q from an empirical transition matrix P(tau),
    using the polynomial adjustment algorithm (Goolsby et al. 2023).

    The core idea: any matrix that is a polynomial in P lives in the same
    eigenspace as P. We find the best valid generator Q = f(P) (a polynomial
    in P) that approximates logm(P)/tau, subject to the constraints that make
    Q a proper rate matrix.

    In this implementaion, by defualt we estimate Q_target using diagonaladjustment
    """

    # ------------------------------------------------------------------
    # Step 1: Normalize P to ensure it is a proper row-stochastic matrix
    # ------------------------------------------------------------------
    P = _row_normalize(P)
    d = P.shape[0]

    # ------------------------------------------------------------------
    # Step 2: Compute target logm(P)/tau
    # This is the "ideal" Q we are trying to approximate.
    # It may not be a valid generator (negative off-diagonals, complex values)
    # which is the entire motivation for the rest of the algorithm.
    # ------------------------------------------------------------------
    # We consider a few options for taking the log
    if LogMethod == 'safetaylor':
        target_Q = isRealLogTaylor(P, epsilon=epsilon, maxIterations=maxIterations) / tau
    elif LogMethod == 'taylor':
        target_Q = isRealLog(P, epsilon=epsilon, maxIterations=maxIterations) / tau
    elif LogMethod == 'safelog':
        target_Q = _safe_logm(P, tau, symmetrize=symmetrize, epsilon=1.0)
    elif LogMethod == 'pseudo':
        I = np.eye(P.shape[0])
        target_Q = (P - I) / tau
    elif LogMethod == 'diagonaladjustment':
        target_Q = diagonalAdjustment(P, tau=tau, k=0, epsilon=epsilon, maxIterations=maxIterations, LogMethod='pseudo') 

    target_vec = target_Q.flatten(order='F')               # column-major vec, shape (d^2,)

    # ------------------------------------------------------------------
    # Step 3: Build the polynomial span matrix
    #
    # Key insight: if Q = c0*I + c1*P + c2*P^2 + ... + c_{d-1}*P^{d-1},
    # then Q is automatically a polynomial in P and shares its eigenstructure.
    # By Cayley-Hamilton, any polynomial in P of degree >= d collapses into
    # a linear combination of {I, P, ..., P^{d-1}}, so d terms suffice.
    #
    # SpMat[:,i] = vec(P^i) in column-major order, shape (d^2, d)
    # So vec(Q) = SpMat @ c  for coefficient vector c.
    # ------------------------------------------------------------------
    SpMat = np.zeros((d * d, d))
    power = np.eye(d)                   # P^0 = I
    for i in range(d):
        SpMat[:, i] = power.flatten(order='F')
        power = power @ P               # next power

    # ------------------------------------------------------------------
    # Step 4: SVD to find an independent basis for the polynomial space
    #
    # The columns of SpMat may be linearly dependent (e.g. if P has
    # repeated eigenvalues). SVD extracts the truly independent directions
    # and discards the rest via singular value thresholding.
    #
    # SpMat = U @ diag(s) @ Vt
    # We keep only columns of U corresponding to significant singular values.
    # The reduced basis: B = U[:,keep] * s[keep], shape (d^2, D)
    # where D <= d is the effective dimension of the polynomial space.
    # ------------------------------------------------------------------
    U, s, _ = np.linalg.svd(SpMat, full_matrices=False)
    keep = np.abs(s) > thresh
    B = U[:, keep] * s[keep]           # shape (d^2, D)
    D = B.shape[1]
    #print(f"Polynomial basis dimension: {D} (out of {d})")

    # ------------------------------------------------------------------
    # Step 5: Set up the Quadratic Program
    #
    # We want to find coefficients x (shape D,) such that:
    #   B @ x  ~  target_vec
    #
    # This is a least-squares problem:
    #   min_x  ||B @ x - target_vec||^2
    #        = min_x  x.T @ (B.T B) x - 2*(B.T target_vec).T @ x
    #
    # Written in standard QP form:  min  (1/2) x.T P_qp x + q_qp.T x
    #   P_qp = 2 * B.T @ B          (positive definite)
    #   q_qp = -2 * B.T @ target_vec  -> simplified to B.T @ target_vec
    #          (the factor of 2 cancels in the QP solution)
    # ------------------------------------------------------------------
    P_qp = B.T @ B                                        # (D, D)
    q_qp = -(B.T @ target_vec).reshape(D,)               # (D,)

    # ------------------------------------------------------------------
    # Equality constraint: each row of Q must sum to zero (generator property)
    #
    # For a row-stochastic P, any polynomial c0*I + c1*P + ... has each
    # row summing to c0 + c1 + ... = constant. So enforcing ONE row sum = 0
    # forces ALL rows to sum to 0.
    #
    # In column-major vec form, "first row sums to 0" is:
    #   [e1, e1, e1, ...] @ vec(Q) = 0    (e1 = [1,0,0,...])
    # which equals: kron(ones(d), e1) @ B @ x = 0
    # ------------------------------------------------------------------
    e1 = np.zeros(d)
    e1[0] = 1.0
    selector = np.kron(np.ones(d), e1)   # shape (d^2,), picks first element of each column block
    A_eq = (selector @ B).reshape(1, D)  # (1, D)
    b_eq = np.zeros(1)

    # ------------------------------------------------------------------
    # Inequality constraint: off-diagonal entries of Q must be >= 0
    #
    # The off-diagonal mask selects all (i,j) with i != j.
    # In vec form: off_diag_rows @ vec(Q) >= 0
    #            -> -(off_diag_rows @ B) @ x <= 0
    # ------------------------------------------------------------------
    diag_mask = np.eye(d).flatten(order='F')              # 1 on diagonal positions
    off_diag_rows = np.eye(d * d)[diag_mask == 0, :]      # (d^2-d, d^2) selects off-diagonals
    G_ineq = -(off_diag_rows @ B)                         # (d^2-d, D)
    h_ineq = np.zeros(G_ineq.shape[0])

    # ------------------------------------------------------------------
    # Step 6: Solve the QP
    # ------------------------------------------------------------------
    # ['daqp', 'osqp', 'proxqp', 'scs']
    x_hat = solve_qp(P_qp, q_qp, G_ineq, h_ineq, A_eq, b_eq, verbose=False, solver='quadprog') #quadprog')

    if x_hat is None:
        raise RuntimeError("QP solver failed to find a solution.")

    # ------------------------------------------------------------------
    # Step 7: Reconstruct Q from the solution coefficients
    # vec(Q) = B @ x_hat  ->  reshape back to (d, d)
    # ------------------------------------------------------------------
    Q = (B @ x_hat).reshape(d, d, order='F')

    # ------------------------------------------------------------------
    # Step 8: Minor cleanup
    # Any residual numerically negative off-diagonal values (should be
    # tiny, e.g. -1e-15) are clamped to zero.
    # ------------------------------------------------------------------
    mask = ~np.eye(d, dtype=bool)
    Q[mask] = np.maximum(Q[mask], 0.0)

    return Q

#######################################
# The functions below are for test 
# purposes
#######################################
## polySolver4: This is just for tests
# if you have pi or sure the 
# dynamics is reversible
#######################################
def polySolver4(C, tau, pi=None, LogMethod='pseudo', floor=1e-12):
    """Polynomial adjustment with/w/o pi but assuming the dynamics is reversible

    1) For P with distinct eigenvalues, {polynomials in P} is EXACTLY
    {matrices with the same eigenvectors as P}. Lagrange interpolation gives a
    polynomial realising any assignment of eigenvalues, so nothing is lost by
    parametrising Q = V diag(mu) V^-1 with mu free. The Cayley-Hamilton basis
    {I, P, ..., P^(d-1)} spans the same space but is catastrophically
    ill-conditioned (powers of P converge to the stationary matrix), so SVD
    truncation silently deletes the fast modes.

    2) If the chain is reversible and pi is known,
    S = Pi^1/2 P Pi^-1/2 is symmetric: real eigenvalues, orthogonal
    eigenvectors, no complex arithmetic anywhere. And in the pi-weighted
    Frobenius norm  ||A||_pi = ||Pi^1/2 A Pi^-1/2||_F  the objective
    DECOUPLES exactly into sum_k (mu_k - log(lambda_k)/tau)^2.

    Constraints are the same: rows of Q sum to zero, off-diagonals >= 0.
    """
    C = np.asarray(C, float)
    if pi is None:
        P = C / C.sum(axis=1, keepdims=True)
        w, U = np.linalg.eig(P.T)
        pi = np.abs(np.real(U[:, np.argmin(np.abs(w - 1.0))]))   # assuming the dynamics is reversible
    return polySolver_reversible(C,tau, pi, LogMethod=LogMethod, floor=floor)

def _qp(target, B, off_idx, mu0=None):
    """min ||mu - target||^2 s.t. (B @ mu)[off_idx] >= 0, with mu[0] fixed at 0.

    B maps eigenvalue coefficients mu to vec(Q). 
    """
    # We use a qp_solver
    A = B[off_idx][:, 1:]                 # (K^2 - K, K-1)
    t = np.asarray(target[1:], float)
    d = t.size
    x = solve_qp(P=2.0 * np.eye(d), q=-2.0 * t,
                 G=-A, h=np.zeros(A.shape[0]), solver="quadprog")
    return np.concatenate(([0.0], x)), None

def _spectral_target(lam, tau, LogMethod="log", floor=1e-12, order=64, delta=None):
    """We use a few options to calculate log (applied to eigenvalues and not the matrix.
    The goal is to avoud diverges."""
    lam = np.asarray(lam, float)
    if LogMethod == "log":
        return np.log(np.clip(lam, floor, None)) / tau
    if LogMethod == "pseudo":
        return (lam - 1.0) / tau
    if LogMethod == "taylor":
        x = lam - 1.0
        return sum((-1)**(n+1) * x**n / n for n in range(1, order + 1)) / tau
    if LogMethod == "hybrid":      # log above the noise floor, pseudo below
        return np.where(lam > delta, np.log(np.clip(lam, floor, None)), lam - 1.0) / tau
    raise ValueError(LogMethod)

def polySolver_reversible(C, tau, pi, LogMethod='pseudo', floor=1e-12):
    """Polynomial adjustment for a reversible dynamics.
    """
    C = np.asarray(C, float)
    P = C / C.sum(axis=1, keepdims=True)
    pi = np.asarray(pi, float)
    pi = pi / pi.sum()
    s = np.sqrt(pi)

    # symmetrise in the pi-metric, then force exact symmetry (this is the
    # reversible projection of P-hat -- also removes sampling asymmetry)
    S = (s[:, None] * P) / s[None, :]
    S = 0.5 * (S + S.T)
    lam, U = np.linalg.eigh(S)

    order = np.argsort(lam)[::-1]
    lam, U = lam[order], U[:, order]

    # target eigenvalues of Q. lam <= 0 carries no usable timescale: aim it at
    # a very fast decay and let the QP pull it back if the constraints bind.
    #target = np.log(np.clip(lam, floor, None)) / tau
    target = _spectral_target(lam, tau, LogMethod=LogMethod, floor=floor)
    target[0] = 0.0

    # basis: vec(Q) = sum_k mu_k * vec(Pi^-1/2 u_k u_k^T Pi^1/2)
    K = len(pi)
    B = np.empty((K * K, K))
    for k in range(K):
        M = np.outer(U[:, k], U[:, k])
        B[:, k] = ((M / s[:, None]) * s[None, :]).ravel()

    eye = np.eye(K, dtype=bool).ravel()
    mu, res = _qp(target, B, ~eye, mu0=target)
    Q = (B @ mu).reshape(K, K)

    Q[~np.eye(K, dtype=bool)] = np.maximum(Q[~np.eye(K, dtype=bool)], 0.0)
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q#, {"lam": lam, "mu": mu, "target": target, "ok": res.success}

# Why do we use below? Since our count matrix sum to very large values!
# L = - 1 / N \sum C_ij log P_ij + (lambda/N) * R
def _nll_gradNEW(logD, F, C, dx, kbT, tau, lam):
    """Negative log-likelihood PER COUNT and its exact gradient w.r.t. log D."""
    D = np.exp(logD)
    K = len(F)
    i = np.arange(K - 1)
    Q, w, U = _eig(D, F, dx, kbT)
    a = w * tau
    ratio = np.exp((F[:, None] - F[None, :]) / (2 * kbT))
    P = np.maximum(((U * np.exp(a)) @ U.T) * ratio, 1e-30)

    # Normalize likelihood by total counts
    N = max(C.sum(), 1.0)
    m = C > 0
    nll = -np.sum(C[m] * np.log(P[m])) / N

    B = np.zeros_like(P)
    B[m] = -C[m] / P[m]
    B *= ratio

    # derivative of a symmetric matrix exponential (Daleckii-Krein):
    # W[m,n] = (e^a_m - e^a_n) / (a_m - a_n), written to avoid overflow
    dd = np.abs(a[:, None] - a[None, :])
    small = dd < 1e-8
    safe = np.where(small, 1.0, dd)
    W = np.exp(np.maximum(a[:, None], a[None, :])) * np.where(
        small, 1.0 - 0.5 * dd, -np.expm1(-safe) / safe)
    R = U @ (U.T @ B @ U * W).T @ U.T

    grad = tau * (-R[i, i] * Q[i, i + 1] - R[i + 1, i + 1] * Q[i + 1, i]
                  + (R[i, i + 1] + R[i + 1, i]) * D / dx**2)
    # smoothing
    if lam > 0:
        L = np.diff(np.eye(K - 1), n=2, axis=0)
        nll += lam * np.sum(np.diff(logD, n=2) ** 2)
        grad += 2 * lam * (L.T @ L @ logD)
    return nll, grad

#######################################
## Validators
#######################################
def validateGenerator(Q, tol=1e-10):
    """Check that Q is a finite, irreducible row generator."""
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError("Rate matrix must be square.")
    if not np.all(np.isfinite(Q)):
        raise ValueError("Rate matrix contains non-finite values.")

    scale = max(1.0, np.max(np.abs(Q)))
    atol = tol * scale
    if np.max(np.abs(Q.sum(axis=1))) > atol:
        raise ValueError("Rate-matrix rows do not sum to zero.")
    if np.any(np.diag(Q) > atol):
        raise ValueError("Rate matrix has a positive diagonal entry.")

    offdiag = Q.copy()
    np.fill_diagonal(offdiag, 0.0)
    if np.any(offdiag < -atol):
        raise ValueError("Rate matrix has a negative off-diagonal entry.")

    adjacency = offdiag > 0.0
    def reachable(graph, start=0):
        seen = {start}
        stack = [start]
        while stack:
            state = stack.pop()
            for other in np.where(graph[state])[0]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
        return len(seen) == graph.shape[0]

    if not reachable(adjacency) or not reachable(adjacency.T):
        raise ValueError("Rate matrix is reducible; not all states communicate.")
    return Q

def db_violation(Q, pi=None):
    """Mean relative detailed-balance violation on the NEAREST-NEIGHBOUR band.

    PA does not impose reversibility, so this is a free check. Restricted to
    the band because far-off-band entries are clamped to exactly zero by the
    inequality constraint, which saturates any relative measure at 2.
    """
    p = pi_from_Q(Q) if pi is None else np.asarray(pi, float) / np.sum(pi)
    f = p[:, None] * Q
    i = np.arange(len(Q) - 1)
    a, b = f[i, i + 1], f[i + 1, i]
    scale = 0.5 * (np.abs(a) + np.abs(b))
    m = scale > 0
    if not m.any():
        return 0.0
    return float(np.mean(np.abs(a - b)[m] / scale[m]))


def tridiagonal_leakage(Q):
    """Fraction of off-diagonal rate mass outside the nearest-neighbour band."""
    K = len(Q)
    off = Q.copy()
    np.fill_diagonal(off, 0.0)
    band = np.zeros_like(off, dtype=bool)
    i = np.arange(K - 1)
    band[i, i + 1] = band[i + 1, i] = True
    return float(off[~band].sum() / off.sum())
