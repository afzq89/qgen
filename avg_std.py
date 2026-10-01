import os
import numpy as np

def aggregate_results(results_list):
    """Combine bootstrap dictionaries into key -> list of bootstrap values."""
    agg = {}
    for result in results_list:
        for key, value in result.items():
            agg.setdefault(key, []).append(value)
    return agg


def profile_keep_mask(profiles, threshold=1.5, min_cases=4, log_space=False):
    """
    Return a bootstrap mask for FE- or D-like profiles using an IQR filter.

    Each bootstrap profile is reduced to one RMS-distance score relative to
    the median profile. Because a larger score means a more unusual profile,
    only scores above Q3 + threshold * IQR are removed.

    ``threshold`` is the IQR multiplier; 1.5 is the standard Tukey rule.
    """
    profiles = np.asarray(profiles, dtype=float)
    if profiles.ndim == 1:
        profiles = profiles[None, :]

    finite = np.all(np.isfinite(profiles), axis=1)
    if log_space:
        finite &= np.all(profiles > 0, axis=1)

    keep = np.zeros(len(profiles), dtype=bool)
    valid = profiles[finite]

    if len(valid) < min_cases:
        keep[finite] = True
        return keep

    if log_space:
        valid = np.log(valid)

    median_profile = np.median(valid, axis=0)
    score = np.sqrt(np.mean((valid - median_profile) ** 2, axis=1))

    q1, q3 = np.percentile(score, [25.0, 75.0])
    iqr = q3 - q1

    if iqr <= 0 or not np.isfinite(iqr):
        valid_keep = np.ones(len(valid), dtype=bool)
    else:
        upper_bound = q3 + threshold * iqr
        valid_keep = score <= upper_bound

    keep[np.where(finite)[0]] = valid_keep
    return keep


def scalar_keep_mask(values, threshold=1.5, min_cases=4):
    """
    Return a bootstrap mask for one scalar MFPT or timescale column using IQR.

    Values outside [Q1 - threshold * IQR, Q3 + threshold * IQR] are removed.
    ``threshold`` is the IQR multiplier; 1.5 is the standard Tukey rule.
    """
    values = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(values)
    keep = np.zeros(len(values), dtype=bool)
    valid = values[finite]

    if len(valid) < min_cases:
        keep[finite] = True
        return keep

    q1, q3 = np.percentile(valid, [25.0, 75.0])
    iqr = q3 - q1

    if iqr <= 0 or not np.isfinite(iqr):
        valid_keep = np.ones(len(valid), dtype=bool)
    else:
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        valid_keep = (valid >= lower_bound) & (valid <= upper_bound)

    keep[np.where(finite)[0]] = valid_keep
    return keep


def finite_rows(*arrays):
    """Return rows that are finite in every supplied profile array."""
    keep = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        keep &= np.all(np.isfinite(array), axis=1)
    return keep


def build_linked_outlier_masks(agg_fe, agg_D, threshold=1.5):
    """
    Build independent FE and D profile masks.

    FE tuple expected from rate code:
        (FE_raw, FE_smooth, FE, FE_fixQ)

    D tuple expected from rate code:
        (D_geo, D_geo_s,
         D_jump, D_jump_s,
         D_pot, D_pot_s,
         DM2, DM2_s,
         DM2_pot, DM2_pot_s)

    The raw and smoothed version of the same D use one shared mask so their
    averages are calculated from exactly the same bootstrap samples.
    """
    masks = {}

    for key in set(agg_fe) & set(agg_D):
        fe_values = agg_fe[key]
        D_values = agg_D[key]

        if len(fe_values) != len(D_values):
            raise ValueError(
                f"Bootstrap alignment error for {key}: "
                f"{len(fe_values)} FE cases and {len(D_values)} D cases."
            )

        FE_raw = np.vstack([v[0] for v in fe_values])
        FE = np.vstack([v[2] for v in fe_values])
        FE_fixQ = np.vstack([v[3] for v in fe_values])

        D_geo = np.vstack([v[0] for v in D_values])
        D_geo_s = np.vstack([v[1] for v in D_values])
        D_jump = np.vstack([v[2] for v in D_values])
        D_jump_s = np.vstack([v[3] for v in D_values])
        D_pot = np.vstack([v[4] for v in D_values])
        D_pot_s = np.vstack([v[5] for v in D_values])
        DM2 = np.vstack([v[6] for v in D_values])
        DM2_s = np.vstack([v[7] for v in D_values])
        DM2_pot = np.vstack([v[8] for v in D_values])
        DM2_pot_s = np.vstack([v[9] for v in D_values])

        masks[key] = {
            "FE_raw": profile_keep_mask(FE_raw, threshold),
            "FE": profile_keep_mask(FE, threshold),
            "FE_fixQ": profile_keep_mask(FE_fixQ, threshold),
            "D_geo": profile_keep_mask(D_geo, threshold, log_space=True)
                     & finite_rows(D_geo, D_geo_s),
            "D_jump": profile_keep_mask(D_jump, threshold, log_space=True)
                      & finite_rows(D_jump, D_jump_s),
            "D_pot": profile_keep_mask(D_pot, threshold, log_space=True)
                     & finite_rows(D_pot, D_pot_s),
            "DM2": profile_keep_mask(DM2, threshold, log_space=True)
                   & finite_rows(DM2, DM2_s),
            "DM2_pot": profile_keep_mask(DM2_pot, threshold, log_space=True)
                       & finite_rows(DM2_pot, DM2_pot_s),
        }

    return masks


def mean_std_profile(values, keep):
    """Calculate profile mean/std; return NaNs if no bootstrap remains."""
    values = np.asarray(values, dtype=float)
    if not np.any(keep):
        shape = values.shape[1:]
        return np.full(shape, np.nan), np.full(shape, np.nan)
    return values[keep].mean(axis=0), values[keep].std(axis=0)


def mean_std_scalar(values, keep):
    """Calculate scalar mean/std; return NaNs if no bootstrap remains."""
    values = np.asarray(values, dtype=float)
    if not np.any(keep):
        return np.nan, np.nan
    return values[keep].mean(), values[keep].std()


def compute_avg_std_fe(agg_fe, output_dir, masks):
    """Write average and standard deviation for the FE profiles."""
    for key, values in agg_fe.items():
        if key not in masks:
            continue

        R, tang, lag = key
        FE_raw = np.vstack([v[0] for v in values])
        FE = np.vstack([v[2] for v in values])
        FE_fixQ = np.vstack([v[3] for v in values])

        raw_avg, raw_std = mean_std_profile(FE_raw, masks[key]["FE_raw"])
        fe_avg, fe_std = mean_std_profile(FE, masks[key]["FE"])
        fix_avg, fix_std = mean_std_profile(FE_fixQ, masks[key]["FE_fixQ"])

        output = os.path.join(output_dir, f"fe_{R:.2f}_{tang:.2f}_{lag}.txt")
        np.savetxt(
            output,
            np.column_stack((raw_avg, raw_std, fe_avg, fe_std, fix_avg, fix_std)),
            fmt="%.6f",
            header=(
                f"# FE_raw used {masks[key]['FE_raw'].sum()} of {len(values)}\n"
                f"# FE used {masks[key]['FE'].sum()} of {len(values)}\n"
                f"# FE_fixQ used {masks[key]['FE_fixQ'].sum()} of {len(values)}\n"
                "# FE_raw std FE std FE_fixQ std"
            ),
            comments="",
        )


def compute_avg_std_D(agg_D, output_dir, masks):
    """Write average and standard deviation for all D profiles."""
    for key, values in agg_D.items():
        if key not in masks:
            continue

        R, tang, lag = key
        arrays = [np.vstack([v[i] for v in values]) for i in range(10)]

        names_and_masks = [
            ("D_geo", 0, 1),
            ("D_jump", 2, 3),
            ("D_pot", 4, 5),
            ("DM2", 6, 7),
            ("DM2_pot", 8, 9),
        ]

        columns = []
        count_lines = []
        for name, raw_i, smooth_i in names_and_masks:
            keep = masks[key][name]
            raw_avg, raw_std = mean_std_profile(arrays[raw_i], keep)
            smooth_avg, smooth_std = mean_std_profile(arrays[smooth_i], keep)
            columns.extend((raw_avg, raw_std, smooth_avg, smooth_std))
            count_lines.append(f"# {name} used {keep.sum()} of {len(values)}")

        output = os.path.join(output_dir, f"D_{R:.2f}_{tang:.2f}_{lag}.txt")
        np.savetxt(
            output,
            np.column_stack(columns),
            fmt="%.6f",
            header=(
                "\n".join(count_lines)
                + "\n# D_geo std D_geo_s std "
                  "D_jump std D_jump_s std "
                  "D_pot std D_pot_s std "
                  "DM2 std DM2_s std "
                  "DM2_pot std DM2_pot_s std"
            ),
            comments="",
        )


def compute_mfpt(agg_mfpt, output_file, masks, threshold=1.5):
    """
    Apply the requested linked masks, followed by a separate scalar-outlier
    test for every MFPT/timescale column.

    MFPT tuple indices:
      0 mfpt_geo       -> FE + D_geo
      1 mfpt_fixQ      -> FE_fixQ + DM2
      2 mfpt_poi       -> direct scalar filtering
      3 mfpt_jum       -> FE + D_jump
      4 mpft_pot       -> FE + D_pot
      5 mfpt5          -> FE_fixQ + DM2_pot
      6 bmfpt_geo      -> FE + D_geo
      7 bmfpt_fixQ     -> FE_fixQ + DM2
      8 b_mfpt_poi     -> direct scalar filtering
      9 sl1            -> direct scalar filtering
     10 sl2            -> direct scalar filtering
     11 sl3            -> direct scalar filtering
    """
    dependencies = {
        0: ("FE", "D_geo"),
        1: ("FE_fixQ", "DM2"),
        2: (),
        3: ("FE", "D_jump"),
        4: ("FE", "D_pot"),
        5: ("FE_fixQ", "DM2_pot"),
        6: ("FE", "D_geo"),
        7: ("FE_fixQ", "DM2"),
        8: (),
        9: (),
        10: (),
        11: (),
    }

    rows = []
    for key, values in agg_mfpt.items():
        if key not in masks:
            continue

        R, tang, lag = key
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.shape[1] != 12:
            raise ValueError(f"Expected 12 MFPT/timescale columns for {key}; got {values.shape[1]}.")

        n_boot = len(values)
        summary = []

        for column in range(12):
            keep = scalar_keep_mask(values[:, column], threshold)
            for dependency in dependencies[column]:
                dependency_mask = masks[key][dependency]
                if len(dependency_mask) != n_boot:
                    raise ValueError(f"Mask length mismatch for {key}, {dependency}.")
                keep &= dependency_mask

            avg, std = mean_std_scalar(values[:, column], keep)
            summary.extend((avg, std))

        rows.append([R, tang, lag, *summary])

    if not rows:
        print("No MFPT data found; output was not written.")
        return

    rows = np.asarray(rows, dtype=float)
    rows = rows[np.lexsort((rows[:, 2], rows[:, 1], rows[:, 0]))]

    header = (
        "#R tang lag "
        "mfpt_geo std "
        "mfpt_fixQ std "
        "mfpt_poi std "
        "mfpt_jum std "
        "mpft_pot std "
        "mfpt5 std "
        "bmfpt_geo std "
        "bmfpt_fixQ std "
        "b_mfpt_poi std "
        "sl1 std "
        "sl2 std "
        "sl3 std"
    )

    fmt = ["%.2f", "%.2f", "%d"] + ["%.3f", "%.3f"] * 12
    np.savetxt(output_file, rows, fmt=fmt, header=header, comments="")
