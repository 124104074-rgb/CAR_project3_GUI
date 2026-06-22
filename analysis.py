import numpy as np
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
from datetime import timedelta

# ==========================================
# SYSTEM PARAMETERS
# ==========================================

PRX_WINDOW_SAMPLES = 60

CPP_BIN_WIDTH = 5
CPP_MIN = 40
CPP_MAX = 120

CPP_OPT_HISTORY = 240

MIN_R2 = 0.20
MIN_BINS_REQUIRED = 5
MIN_VALUES_PER_BIN = 3
MAX_ALLOWED_MIN_PRX = 0.25

MAX_ALLOWED_GAP_SEC = 10

REAL_SAMPLE_INTERVAL = 5.0

# ==========================================
# CPP CALCULATION
# ==========================================

def calculate_cpp(map_value, icp_value):
    return map_value - icp_value


# ==========================================
# PRX CALCULATION
# ==========================================

def calculate_prx(map_window, icp_window):

    if len(map_window) < PRX_WINDOW_SAMPLES:
        return np.nan, f"Need {PRX_WINDOW_SAMPLES - len(map_window)} more samples"

    try:
        prx, _ = pearsonr(map_window, icp_window)
        return prx, "Valid"

    except Exception as e:
        return np.nan, f"Correlation error : {str(e)}"


# ==========================================
# mean CPP CALCULATION
# ==========================================

def calculate_mean_cpp(cpp_window):

    if len(cpp_window) < PRX_WINDOW_SAMPLES:
        return np.nan

    return np.mean(cpp_window)

# ==========================================
# QUADRATIC FUNCTION
# ==========================================

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


# ==========================================
# R² CALCULATION
# ==========================================

def calculate_r2(y_actual, y_pred):

    ss_res = np.sum((y_actual - y_pred) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)

    if ss_tot == 0:
        return 0

    return 1 - (ss_res / ss_tot)


# ==========================================
# CPPopt CALCULATION
# ==========================================
def calculate_cppopt(prx_hist, cpp_hist):

    if len(prx_hist) < CPP_OPT_HISTORY:
        return (
            np.nan,
            f"Need {CPP_OPT_HISTORY - len(prx_hist)} more PRx values"
        )

    prx_hist = np.array(prx_hist)[-CPP_OPT_HISTORY:]  # Last 240 PRx values
    cpp_hist = np.array(cpp_hist)[-CPP_OPT_HISTORY:]  # Last 240 mean CPP values

    valid_mask = (                                   # Remove NaN values
        ~np.isnan(prx_hist)
        &
        ~np.isnan(cpp_hist)
    )

    prx_hist = prx_hist[valid_mask]                  # Keep valid PRx values
    cpp_hist = cpp_hist[valid_mask]                  # Keep valid CPP values

    if len(prx_hist) < MIN_BINS_REQUIRED * MIN_VALUES_PER_BIN:
        return (
            np.nan,
            "Not enough valid data"
        )

    bins = np.arange(
        cpp_hist.min(),
        cpp_hist.max() + CPP_BIN_WIDTH,
        CPP_BIN_WIDTH
    )

    bin_centers = []
    bin_prx = []

    for i in range(len(bins) - 1):

        mask = (
            (cpp_hist >= bins[i])
            &
            (cpp_hist < bins[i + 1])
        )

        if np.sum(mask) >= MIN_VALUES_PER_BIN:

            bin_centers.append(
                (bins[i] + bins[i + 1]) / 2
            )

            bin_prx.append(
                np.mean(prx_hist[mask])
            )

    if len(bin_centers) < MIN_BINS_REQUIRED:
        return (
            np.nan,
            f"Need at least {MIN_BINS_REQUIRED} CPP bins"
        )

    bin_centers = np.array(bin_centers)
    bin_prx = np.array(bin_prx)

    try:

        popt, _ = curve_fit(
            quadratic,
            bin_centers,
            bin_prx
        )

        a, b, c = popt

        if a <= 0:
            return np.nan, "Quadratic coefficient a <= 0"

        cppopt = -b / (2 * a)

        if cppopt < CPP_MIN or cppopt > CPP_MAX:
            return (
                np.nan,
                f"CPPopt outside range ({cppopt:.1f})"
            )

        fitted = quadratic(bin_centers, *popt)

        r2 = calculate_r2(bin_prx, fitted)

        if r2 < MIN_R2:
            return np.nan, f"Poor fit R²={r2:.2f}"

        min_prx = quadratic(cppopt, *popt)

        if min_prx > MAX_ALLOWED_MIN_PRX:
            return (
                np.nan,
                f"Minimum PRx too high ({min_prx:.2f})"
            )

        return cppopt, "Valid"

    except Exception as e:

        return (
            np.nan,
            f"Curve fitting error : {str(e)}"
        )
# ==========================================
# GAP DETECTION
# ==========================================

def check_gap(prev_time, current_time):

    if prev_time is None:
        return False, 0

    gap_sec = (
        current_time - prev_time
    ).total_seconds()

    if gap_sec > MAX_ALLOWED_GAP_SEC:
        return True, gap_sec

    return False, gap_sec
