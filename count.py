import numpy as np

# Inspired by deeptime

def countMatrix(dtrajs, lag, n_states=None, mode="sliding", stride=None):
    """
    Count matrix from discrete trajectories.
    """
    if lag < 1:
        raise ValueError("lag must be >= 1")

    if n_states is None:
        valid_trajs = [np.asarray(t, dtype=int) for t in dtrajs if len(t) > 0]
        if not valid_trajs:
            raise ValueError("No valid trajectories found.")
        n_states = max(np.max(t) for t in valid_trajs) + 1

    C = np.zeros((n_states, n_states), dtype=float)

    for traj in dtrajs:
        traj = np.asarray(traj, dtype=int)

        if len(traj) <= lag:
            continue

        if mode == "sliding" or mode == "sliding-effective":
            starts = traj[:-lag]
            ends   = traj[lag:]

        elif mode == "sample":
            idx = np.arange(0, len(traj) - lag, lag)
            starts = traj[idx]
            ends   = traj[idx + lag]

        elif mode == "effective":
            if stride is None:
                stride = lag
            if stride < 1:
                raise ValueError("stride must be >= 1")
            idx = np.arange(0, len(traj) - lag, stride)
            starts = traj[idx]
            ends   = traj[idx + lag]

        else:
            raise ValueError(f"Unknown mode: {mode}")

        np.add.at(C, (starts, ends), 1.0)

    if mode == "sliding-effective":
        C /= float(lag)

    return C

def estimate_correlation_length(traj, max_lag=None):
    traj = np.array(traj, dtype=float)  # ensure float for mean subtraction
    traj = traj - traj.mean()
    N = len(traj)

    # if trajectory is constant, no correlation structure → return 1
    if np.all(traj == 0):
        return 1

    if max_lag is None:
        max_lag = N // 3

    acf_full = np.fft.ifft(np.abs(np.fft.fft(traj, n=2*N))**2).real[:N]

    # guard against zero variance
    if acf_full[0] == 0:
        return 1

    acf = acf_full / acf_full[0]

    tau = 0.5
    for k in range(1, max_lag):
        if acf[k] < 1/np.e:
            break
        tau += acf[k]

    return int(np.ceil(2 * tau))

def estimate_correlation_length_from_trajs(trajs):
    taus = [estimate_correlation_length(t) for t in trajs]
    return max(taus)

def bootstrap_trajectories(trajs, correlation_length, rng=None):
    '''
    Block bootstrapping for time-series trajectories.
    '''
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(trajs[0], (int, float)):
        trajs = [trajs]

    lengths = np.array([len(t) for t in trajs])
    Ltot = lengths.sum()
    n_trajs = len(trajs)

    if correlation_length < 1:
        correlation_length = lengths.max()

    w = lengths / Ltot

    # Estimate blocks needed. If data is heavily fragmented (-1 splits),
    # average trajectory length will be small, so we'll need more blocks.
    expected_block_size = min(correlation_length, lengths.mean())
    batch_size = int(np.ceil(Ltot / expected_block_size) * 1.2) # 20% safety buffer

    subs = []
    Laccum = 0

    while Laccum < Ltot:

        chosen_itrajs = rng.choice(n_trajs, size=batch_size, p=w)
        chosen_lengths = lengths[chosen_itrajs]

        highs = np.maximum(1, chosen_lengths - correlation_length)

        t0s = rng.integers(0, highs)
        t1s = np.minimum(chosen_lengths, t0s + correlation_length)

        block_lengths = t1s - t0s
        cum_lengths = np.cumsum(block_lengths)

        remaining_L = Ltot - Laccum
        # Find the exact array index where we hit our target length
        stop_idx = np.searchsorted(cum_lengths, remaining_L) + 1

        if stop_idx > batch_size:
            stop_idx = batch_size

        subs.extend([
            trajs[chosen_itrajs[i]][t0s[i]:t1s[i]]
            for i in range(stop_idx)
        ])

        Laccum += cum_lengths[stop_idx - 1]

    return subs
