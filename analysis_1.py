import numpy as np  # Import numerical operations
from scipy.stats import pearsonr  # Import Pearson correlation for PRx
from scipy.optimize import curve_fit  # Import curve fitting for CPPopt

PRX_WINDOW_SAMPLES = 60  # Use 60 samples = 5 minutes at one sample every 5 seconds
CPP_BIN_WIDTH = 5  # Use 5 mmHg CPP bins for CPPopt
CPP_MIN = 40  # Reject CPPopt below 40 mmHg
CPP_MAX = 120  # Reject CPPopt above 120 mmHg
CPP_OPT_HISTORY = 240  # Use 240 PRx-mean CPP pairs = 4 hours at one pair per minute
MIN_R2 = 0.20  # Minimum acceptable quadratic fit quality
MIN_BINS_REQUIRED = 5  # Require at least five CPP bins
MIN_VALUES_PER_BIN = 3  # Require at least three values in each CPP bin
MAX_ALLOWED_MIN_PRX = 0.25  # Reject CPPopt if fitted minimum PRx is above 0.25
MAX_ALLOWED_GAP_SEC = 10  # Treat gaps greater than 10 seconds as discontinuities
REAL_SAMPLE_INTERVAL = 5.0  # Expected physiological sampling interval in seconds

def calculate_cpp(map_value, icp_value):
    return map_value - icp_value  # Calculate cerebral perfusion pressure

def calculate_prx(map_window, icp_window):
    if len(map_window) < PRX_WINDOW_SAMPLES:  # Check whether the complete PRx window exists
        return np.nan, f"Need {PRX_WINDOW_SAMPLES - len(map_window)} more samples"  # Return waiting status

    map_window = np.asarray(map_window, dtype=float)  # Convert MAP window to numeric array
    icp_window = np.asarray(icp_window, dtype=float)  # Convert ICP window to numeric array

    valid_mask = ~np.isnan(map_window) & ~np.isnan(icp_window)  # Keep only valid paired values
    map_window = map_window[valid_mask]  # Remove invalid MAP values
    icp_window = icp_window[valid_mask]  # Remove invalid ICP values

    if len(map_window) < PRX_WINDOW_SAMPLES:  # Reject incomplete valid windows
        return np.nan, "Invalid or missing values in PRx window"  # Explain why PRx is unavailable

    if np.std(map_window) == 0 or np.std(icp_window) == 0:  # Avoid undefined correlation for constant signals
        return np.nan, "MAP or ICP window has no variation"  # Explain why PRx is unavailable

    try:
        prx, _ = pearsonr(map_window, icp_window)  # Calculate MAP-ICP Pearson correlation
        return prx, "Valid"  # Return valid PRx
    except Exception as e:
        return np.nan, f"Correlation error: {str(e)}"  # Return correlation failure reason

def calculate_mean_cpp(cpp_window):
    if len(cpp_window) < PRX_WINDOW_SAMPLES:  # Check whether the complete CPP window exists
        return np.nan  # Return unavailable mean CPP

    cpp_window = np.asarray(cpp_window, dtype=float)  # Convert CPP window to numeric array

    if np.isnan(cpp_window).any():  # Reject windows containing missing CPP values
        return np.nan  # Return unavailable mean CPP

    return np.mean(cpp_window)  # Return mean CPP from the same window used for PRx

def quadratic(x, a, b, c):
    return a * x**2 + b * x + c  # Define quadratic CPP-PRx relationship

def calculate_r2(y_actual, y_pred):
    ss_res = np.sum((y_actual - y_pred) ** 2)  # Calculate residual sum of squares
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)  # Calculate total sum of squares

    if ss_tot == 0:  # Prevent division by zero for constant data
        return 0  # Treat constant data as an unacceptable fit

    return 1 - (ss_res / ss_tot)  # Return coefficient of determination

def calculate_cppopt(prx_hist, cpp_hist):
    if len(prx_hist) < CPP_OPT_HISTORY or len(cpp_hist) < CPP_OPT_HISTORY:  # Require full 4-hour history
        available_pairs = min(len(prx_hist), len(cpp_hist))  # Find available paired history length
        return np.nan, f"Need {CPP_OPT_HISTORY - available_pairs} more PRx values"  # Explain waiting status

    prx_hist = np.asarray(prx_hist[-CPP_OPT_HISTORY:], dtype=float)  # Keep latest 240 PRx values
    cpp_hist = np.asarray(cpp_hist[-CPP_OPT_HISTORY:], dtype=float)  # Keep latest 240 mean CPP values

    valid_mask = ~np.isnan(prx_hist) & ~np.isnan(cpp_hist)  # Identify valid PRx-CPP pairs

    if np.sum(valid_mask) < CPP_OPT_HISTORY:  # Require all 240 pairs to be valid for strict 4-hour CPPopt
        invalid_count = CPP_OPT_HISTORY - np.sum(valid_mask)  # Count invalid pairs
        return np.nan, f"{invalid_count} invalid PRx-CPP pairs in 4-hour window"  # Explain rejection

    prx_hist = prx_hist[valid_mask]  # Keep valid PRx values
    cpp_hist = cpp_hist[valid_mask]  # Keep valid mean CPP values

    bins = np.arange(cpp_hist.min(), cpp_hist.max() + CPP_BIN_WIDTH, CPP_BIN_WIDTH)  # Create CPP bins

    if len(bins) < 2:  # Check whether CPP has enough range for binning
        return np.nan, "CPP range is too small for CPPopt calculation"  # Explain rejection

    bin_centers = []  # Store CPP-bin centres
    bin_prx = []  # Store mean PRx for each CPP bin

    for i in range(len(bins) - 1):  # Visit each CPP bin
        if i == len(bins) - 2:  # Handle final bin separately so maximum CPP is included
            mask = (cpp_hist >= bins[i]) & (cpp_hist <= bins[i + 1])  # Include upper boundary in final bin
        else:
            mask = (cpp_hist >= bins[i]) & (cpp_hist < bins[i + 1])  # Select values in normal bin

        if np.sum(mask) >= MIN_VALUES_PER_BIN:  # Keep only adequately populated bins
            bin_centers.append((bins[i] + bins[i + 1]) / 2)  # Store midpoint of CPP bin
            bin_prx.append(np.mean(prx_hist[mask]))  # Store mean PRx in CPP bin

    if len(bin_centers) < MIN_BINS_REQUIRED:  # Require enough distinct CPP bins
        return np.nan, f"Need at least {MIN_BINS_REQUIRED} populated CPP bins"  # Explain rejection

    bin_centers = np.asarray(bin_centers, dtype=float)  # Convert bin centres to numeric array
    bin_prx = np.asarray(bin_prx, dtype=float)  # Convert bin PRx values to numeric array

    try:
        popt, _ = curve_fit(quadratic, bin_centers, bin_prx)  # Fit quadratic curve to binned data
        a, b, c = popt  # Extract quadratic coefficients

        if a <= 0:  # CPPopt requires a U-shaped curve
            return np.nan, "No U-shaped CPP-PRx curve"  # Explain rejection

        cppopt = -b / (2 * a)  # Calculate quadratic minimum

        if cppopt < CPP_MIN or cppopt > CPP_MAX:  # Reject physiologically implausible CPPopt
            return np.nan, f"CPPopt outside range ({cppopt:.1f} mmHg)"  # Explain rejection

        fitted_prx = quadratic(bin_centers, *popt)  # Calculate fitted PRx values
        r2 = calculate_r2(bin_prx, fitted_prx)  # Calculate fit quality

        if r2 < MIN_R2:  # Reject weak quadratic fit
            return np.nan, f"Poor fit R² = {r2:.2f}"  # Explain rejection

        min_prx = quadratic(cppopt, *popt)  # Calculate PRx at fitted minimum

        if min_prx > MAX_ALLOWED_MIN_PRX:  # Reject if autoregulation is poor even at fitted optimum
            return np.nan, f"Minimum PRx too high ({min_prx:.2f})"  # Explain rejection

        return cppopt, "Valid"  # Return valid CPPopt

    except Exception as e:
        return np.nan, f"Curve fitting error: {str(e)}"  # Return curve-fitting failure reason

def check_gap(prev_time, current_time):
    if prev_time is None:  # No previous timestamp exists for first sample
        return False, 0  # No gap exists

    gap_sec = (current_time - prev_time).total_seconds()  # Calculate elapsed time between samples

    if gap_sec > MAX_ALLOWED_GAP_SEC:  # Detect discontinuity larger than allowed threshold
        return True, gap_sec  # Return detected gap

    return False, gap_sec  # Return normal interval