import streamlit as st  # Import Streamlit for dashboard interface
import pandas as pd  # Import pandas for CSV reading and graph tables
import numpy as np  # Import NumPy for NaN handling and numerical operations
import time  # Import time for controlled dashboard refresh
from datetime import datetime, timedelta  # Import datetime tools for monitor timestamps and gap points

try:
    import plotly.graph_objects as go  # Import Plotly for colored ICU-style graphs
    HAVE_PLOTLY = True  # Store Plotly availability
except Exception:
    HAVE_PLOTLY = False  # Fall back to Streamlit line_chart if Plotly is not installed

from analysis_1 import (  # Import analysis functions and constants from analysis.py
    calculate_cpp,  # Import CPP calculation function
    calculate_prx,  # Import PRx calculation function
    calculate_mean_cpp,  # Import mean CPP calculation function
    calculate_cppopt,  # Import CPPopt calculation function
    check_gap,  # Import timestamp gap detection function
    PRX_WINDOW_SAMPLES,  # Import PRx rolling-window sample count
    REAL_SAMPLE_INTERVAL  # Import expected real monitor sample interval
)

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(page_title="CPPopt Monitor", layout="wide")  # Configure Streamlit browser page

# =========================================================
# SYSTEM PARAMETERS
# =========================================================

DATA_FILE = "live_data.csv"  # Define the incoming live CSV file name
PRX_UPDATE_INTERVAL_SEC = 60  # Define PRx and CPPopt update interval in real monitor seconds
PRX_AUTOREGULATION_THRESHOLD = 0.25  # Define PRx threshold for autoregulation classification
DASHBOARD_REFRESH_SEC = 1.0  # Refresh dashboard once per second
MAX_CPP_GRAPH_POINTS = 50000  # Keep enough CPP points for long graph windows
MAX_PRX_GRAPH_POINTS = 10000  # Keep enough PRx points for long graph windows
MAX_CPPOPT_GRAPH_POINTS = 10000  # Keep enough CPPopt points for long graph windows


CPP_COLOR = "#1f77b4"  # Blue color for CPP graph and CPP value
PRX_COLOR = "#ff7f0e"  # Orange color for PRx graph and PRx value
CPPOPT_COLOR = "#2ca02c"  # Green color for CPPopt graph and CPPopt value
DIFF_COLOR = "#7e57c2"  # Purple color for CPPopt minus CPP value
MAP_COLOR = "#0ea5e9"  # Cyan/blue color for MAP value card
ICP_COLOR = "#ef4444"  # Red color for ICP value card

WINDOW_OPTIONS = {  # Define selectable graph display windows
    "30 min": 30,  # Display last 30 minutes
    "1 hour": 60,  # Display last 1 hour
    "2 hours": 120,  # Display last 2 hours
    "4 hours": 240,  # Display last 4 hours
    "6 hours": 360,  # Display last 6 hours
    "8 hours": 480,  # Display last 8 hours
}

# =========================================================
# CSS STYLE
# =========================================================

st.markdown(
    """
    <style>
    /* Main title: uses Streamlit theme text color, so it stays visible in light and dark mode. */
    .main-title {
        font-size: 42px;
        font-weight: 900;
        color: var(--text-color);
        margin-bottom: 4px;
        line-height: 1.15;
    }

    .subtitle {
        font-size: 18px;
        color: var(--text-color);
        opacity: 0.75;
        margin-bottom: 18px;
    }

    /* Top ICU-style parameter cards. */
    .metric-card {
        border: 1px solid #d0d7de;
        border-radius: 12px;
        padding: 18px 18px;
        background-color: white;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
        min-height: 138px;
    }

    .metric-label {
        font-size: 30px;
        font-weight: 900;
        color: #111827;
        margin-bottom: 8px;
        line-height: 1.05;
    }

    .metric-value {
        font-size: 56px;
        font-weight: 900;
        line-height: 1.0;
        letter-spacing: -1px;
    }

    .metric-sub {
        font-size: 17px;
        color: #475569;
        margin-top: 8px;
        font-weight: 700;
    }

    /* Value cards placed on the right side of each graph. */
    .side-value-card {
        border: 1px solid #d0d7de;
        border-radius: 14px;
        padding: 24px 18px;
        background-color: white;
        min-height: 300px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }

    .side-title {
        font-size: 32px;
        font-weight: 900;
        margin-bottom: 10px;
        line-height: 1.05;
    }

    .side-value {
        font-size: 64px;
        font-weight: 900;
        margin-bottom: 12px;
        line-height: 1.0;
        letter-spacing: -1px;
    }

    .side-caption {
        font-size: 17px;
        color: #475569;
        font-weight: 700;
    }

    .auto-good {
        margin-top: 14px;
        padding: 14px;
        border-radius: 10px;
        background-color: #dcfce7;
        color: #166534;
        font-size: 24px;
        font-weight: 900;
        text-align: center;
        border: 2px solid #22c55e;
    }

    .auto-bad {
        margin-top: 14px;
        padding: 14px;
        border-radius: 10px;
        background-color: #fee2e2;
        color: #991b1b;
        font-size: 24px;
        font-weight: 900;
        text-align: center;
        border: 2px solid #ef4444;
    }

    .auto-na {
        margin-top: 14px;
        padding: 14px;
        border-radius: 10px;
        background-color: #f1f5f9;
        color: #475569;
        font-size: 24px;
        font-weight: 900;
        text-align: center;
        border: 2px solid #94a3b8;
    }

    /* Make Streamlit section headings larger and clearer. */
    h2, h3 {
        color: var(--text-color) !important;
        font-weight: 900 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)  # Add custom CSS for ICU-style cards and larger labels

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def parse_timestamp(row):  # Convert TTDate and TTTime from one CSV row into a datetime object
    date_text = str(row["TTDate"]).strip()  # Read TTDate and remove extra spaces
    time_text = str(row["TTTime"]).strip()  # Read TTTime and remove extra spaces
    timestamp_text = f"{date_text} {time_text}"  # Combine date and time into one timestamp text

    formats = [  # Define possible timestamp formats from monitor exports
        "%d/%m/%Y %H:%M:%S",  # Example: 24/06/2026 13:14:23
        "%d-%m-%Y %H:%M:%S",  # Example: 24-06-2026 13:14:23
        "%Y-%m-%d %H:%M:%S",  # Example: 2026-06-24 13:14:23
        "%m/%d/%Y %H:%M:%S",  # Example: 06/24/2026 13:14:23
        "%Y-%m-%dT%H:%M:%S.%f",  # Example: 2026-06-24T13:14:23.000
        "%Y-%m-%dT%H:%M:%S"  # Example: 2026-06-24T13:14:23
    ]

    for time_format in formats:  # Try each timestamp format one by one
        try:  # Try converting the timestamp text
            return datetime.strptime(timestamp_text, time_format)  # Return datetime when conversion succeeds
        except ValueError:  # Handle an incorrect format without stopping
            pass  # Try the next format

    raise ValueError(f"Cannot parse TTDate and TTTime: {timestamp_text}")  # Stop with a clear timestamp error


def is_nan_value(value):  # Check whether a value is unavailable or NaN
    try:  # Try numeric NaN test
        return value is None or np.isnan(value)  # Return true for None or NaN
    except TypeError:  # Handle non-numeric values safely
        return True  # Treat non-numeric values as unavailable


def format_live_value(value, decimals=2):  # Convert a numeric value into dashboard display text
    if is_nan_value(value):  # Check whether the value is unavailable
        return "Not available"  # Display unavailable text
    return f"{value:.{decimals}f}"  # Display formatted numeric value


def trend_arrow(current_value, previous_value):  # Return trend arrow by comparing current value with previous value
    if is_nan_value(current_value) or is_nan_value(previous_value):  # Check whether trend comparison is possible
        return ""  # Return no arrow when comparison is unavailable
    if current_value > previous_value:  # Check increasing trend
        return "↑"  # Up arrow
    if current_value < previous_value:  # Check decreasing trend
        return "↓"  # Down arrow
    return "→"  # Right arrow for no change


def trim_history():  # Keep graph history sizes limited for stable long-term dashboard performance
    if len(st.session_state.cpp_plot_time) > MAX_CPP_GRAPH_POINTS:  # Check CPP graph length
        st.session_state.cpp_plot_time = st.session_state.cpp_plot_time[-MAX_CPP_GRAPH_POINTS:]  # Keep newest CPP timestamps
        st.session_state.cpp_plot_values = st.session_state.cpp_plot_values[-MAX_CPP_GRAPH_POINTS:]  # Keep newest CPP values

    if len(st.session_state.prx_plot_time) > MAX_PRX_GRAPH_POINTS:  # Check PRx graph length
        st.session_state.prx_plot_time = st.session_state.prx_plot_time[-MAX_PRX_GRAPH_POINTS:]  # Keep newest PRx timestamps
        st.session_state.prx_plot_values = st.session_state.prx_plot_values[-MAX_PRX_GRAPH_POINTS:]  # Keep newest PRx values

    if len(st.session_state.cppopt_plot_time) > MAX_CPPOPT_GRAPH_POINTS:  # Check CPPopt graph length
        st.session_state.cppopt_plot_time = st.session_state.cppopt_plot_time[-MAX_CPPOPT_GRAPH_POINTS:]  # Keep newest CPPopt timestamps
        st.session_state.cppopt_plot_values = st.session_state.cppopt_plot_values[-MAX_CPPOPT_GRAPH_POINTS:]  # Keep newest CPPopt values


def insert_cpp_nan_gap_points(previous_timestamp, current_timestamp):  # Insert NaN points for every expected missing CPP sample
    missing_timestamp = previous_timestamp + timedelta(seconds=REAL_SAMPLE_INTERVAL)  # Start from first missing expected sample time

    while missing_timestamp < current_timestamp:  # Continue until the first real post-gap sample
        st.session_state.cpp_plot_time.append(missing_timestamp)  # Add missing timestamp to CPP graph
        st.session_state.cpp_plot_values.append(np.nan)  # Add NaN so CPP graph line breaks during the gap
        missing_timestamp += timedelta(seconds=REAL_SAMPLE_INTERVAL)  # Move to next expected sample time


def insert_prx_cppopt_nan_gap_points(previous_timestamp, current_timestamp):  # Insert NaN points for PRx and CPPopt during a data gap
    missing_timestamp = previous_timestamp + timedelta(seconds=PRX_UPDATE_INTERVAL_SEC)  # Start from first expected PRx minute point

    while missing_timestamp < current_timestamp:  # Continue until first real post-gap sample
        st.session_state.prx_plot_time.append(missing_timestamp)  # Add missing time to PRx graph
        st.session_state.prx_plot_values.append(np.nan)  # Add NaN to break PRx graph line
        st.session_state.cppopt_plot_time.append(missing_timestamp)  # Add missing time to CPPopt graph
        st.session_state.cppopt_plot_values.append(np.nan)  # Add NaN to break CPPopt graph line
        missing_timestamp += timedelta(seconds=PRX_UPDATE_INTERVAL_SEC)  # Move to next expected minute point


def reset_continuous_windows_after_gap():  # Reset only rolling calculations after a detected timestamp gap
    st.session_state.map_history.clear()  # Remove MAP values before the gap
    st.session_state.icp_history.clear()  # Remove ICP values before the gap
    st.session_state.cpp_history.clear()  # Remove CPP values before the gap
    st.session_state.last_prx_calculation_time = None  # Make next PRx calculation act as a first PRx
    st.session_state.latest_map = np.nan  # Mark live MAP unavailable during gap
    st.session_state.latest_icp = np.nan  # Mark live ICP unavailable during gap
    st.session_state.latest_cpp = np.nan  # Mark live CPP unavailable during gap
    st.session_state.latest_prx = np.nan  # Mark live PRx unavailable during gap
    st.session_state.latest_prx_status = "Waiting for 60 continuous samples after data gap"  # Explain PRx status after gap
    st.session_state.latest_cppopt = np.nan  # Mark current CPPopt unavailable after gap
    st.session_state.latest_cppopt_status = "Waiting for valid PRx history after data gap"  # Explain CPPopt status after gap
    st.session_state.latest_autoregulation_status = "Not available"  # Mark autoregulation unavailable after gap


def crop_to_window(times, values, minutes):  # Return only values within selected graph window
    if len(times) == 0:  # Check whether graph history is empty
        return [], []  # Return empty lists
    latest_time = max(times)  # Use latest available graph timestamp as right edge
    start_time = latest_time - timedelta(minutes=minutes)  # Calculate selected window start time
    filtered_times = []  # Store timestamps inside selected window
    filtered_values = []  # Store values inside selected window
    for t, v in zip(times, values):  # Iterate through graph history
        if t >= start_time:  # Keep only points inside selected window
            filtered_times.append(t)  # Add timestamp
            filtered_values.append(v)  # Add corresponding value
    return filtered_times, filtered_values  # Return filtered graph data


def make_plot_dataframe(times, values, value_name, window_minutes):  # Build plot dataframe for selected window
    filtered_times, filtered_values = crop_to_window(times, values, window_minutes)  # Crop graph history
    return pd.DataFrame({"Time": filtered_times, value_name: filtered_values})  # Return dataframe for plotting


def render_line_chart(plot_df, value_column, color, title):  # Render a graph with selected color
    if plot_df.empty:  # Check whether graph data is available
        st.info(f"Waiting for {title} data...")  # Show waiting message
        return  # Stop graph rendering

    if HAVE_PLOTLY:  # Use Plotly when available for reliable color and gap handling
        fig = go.Figure()  # Create Plotly figure
        fig.add_trace(  # Add one line trace
            go.Scatter(
                x=plot_df["Time"],  # Use full datetime internally
                y=plot_df[value_column],  # Use parameter values
                mode="lines",  # Draw line graph
                line=dict(color=color, width=2.5),  # Set graph color and thickness
                connectgaps=False,  # Do not connect across NaN gaps
                name=value_column,  # Set trace name
            )
        )
        fig.update_layout(  # Configure graph layout
            height=310,  # Set graph height
            margin=dict(l=10, r=10, t=20, b=10),  # Reduce extra margins
            xaxis=dict(tickformat="%H:%M:%S", title="TTTime"),  # Display only TTTime on x-axis
            yaxis=dict(title=value_column),  # Set y-axis label
            showlegend=False,  # Hide legend because graph title is already visible
            template="plotly_white",  # Use clean background
        )
        st.plotly_chart(fig, use_container_width=True)  # Display Plotly chart in Streamlit
    else:  # Fall back to Streamlit chart if Plotly is unavailable
        fallback_df = plot_df.copy()  # Copy dataframe for fallback plotting
        fallback_df["Time"] = pd.to_datetime(fallback_df["Time"]).dt.strftime("%H:%M:%S")  # Show only time labels
        st.line_chart(fallback_df.set_index("Time"), use_container_width=True)  # Draw fallback Streamlit line chart


def render_metric_card(label, value_text, color, arrow="", subtext=""):  # Render a custom ICU-style metric card
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color};">{value_text} {arrow}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )  # Display custom metric card with larger title


def render_side_value(title, value_text, color, arrow="", caption=""):  # Render value card beside each graph
    st.markdown(
        f"""
        <div class="side-value-card">
            <div class="side-title" style="color:{color};">{title}</div>
            <div class="side-value" style="color:{color};">{value_text} {arrow}</div>
            <div class="side-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )  # Display graph-side value card


def render_autoregulation_badge(status):  # Render color-coded autoregulation status badge
    if status == "Preserved autoregulation":  # Check preserved status
        st.markdown('<div class="auto-good">🟢 PRESERVED</div>', unsafe_allow_html=True)  # Show green badge
    elif status == "Impaired autoregulation":  # Check impaired status
        st.markdown('<div class="auto-bad">🔴 IMPAIRED</div>', unsafe_allow_html=True)  # Show red badge
    else:  # Handle unavailable status
        st.markdown('<div class="auto-na">⚪ NOT AVAILABLE</div>', unsafe_allow_html=True)  # Show grey badge

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================

state_defaults = {  # Define all session-state variables and their initial values
    "last_processed_row": 0,  # CSV row index already processed
    "map_history": [],  # MAP history for continuous PRx window
    "icp_history": [],  # ICP history for continuous PRx window
    "cpp_history": [],  # CPP history for mean CPP window
    "prx_history": [],  # Only valid PRx values for CPPopt
    "mean_cpp_history": [],  # Mean CPP values paired with valid PRx
    "cpp_plot_time": [],  # CPP graph timestamps
    "cpp_plot_values": [],  # CPP graph values and NaN gap points
    "prx_plot_time": [],  # PRx graph timestamps
    "prx_plot_values": [],  # PRx graph values and NaN gap points
    "cppopt_plot_time": [],  # CPPopt graph timestamps
    "cppopt_plot_values": [],  # CPPopt graph values and NaN gap points
    "previous_timestamp": None,  # Previous monitor timestamp for gap detection
    "last_prx_calculation_time": None,  # Last PRx attempt timestamp
    "latest_map": np.nan,  # Latest MAP value
    "latest_icp": np.nan,  # Latest ICP value
    "latest_cpp": np.nan,  # Latest CPP value
    "latest_prx": np.nan,  # Latest PRx value
    "latest_cppopt": np.nan,  # Latest CPPopt value
    "previous_map": np.nan,  # Previous MAP value for trend arrow
    "previous_icp": np.nan,  # Previous ICP value for trend arrow
    "previous_cpp": np.nan,  # Previous CPP value for trend arrow
    "previous_prx": np.nan,  # Previous PRx value for trend arrow
    "previous_cppopt": np.nan,  # Previous CPPopt value for trend arrow
    "previous_cppopt_minus_cpp": np.nan,  # Previous CPPopt minus CPP for trend arrow
    "latest_prx_status": "Waiting for 60 continuous samples",  # PRx availability reason
    "latest_cppopt_status": "Waiting for valid PRx values",  # CPPopt availability reason
    "last_valid_cppopt": np.nan,  # Last valid CPPopt value
    "last_valid_cppopt_time": None,  # Timestamp of last valid CPPopt
    "latest_autoregulation_status": "Not available",  # Autoregulation interpretation
    "latest_gap_message": "",  # Latest gap warning message
    "latest_monitor_timestamp": None,  # Latest TTDate + TTTime timestamp
}

for key, value in state_defaults.items():  # Initialize each required session-state item
    if key not in st.session_state:  # Create item only if not already present
        st.session_state[key] = value.copy() if isinstance(value, list) else value  # Avoid shared list references

# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.header("Graph Window")  # Display sidebar heading for graph windows
cpp_window_label = st.sidebar.selectbox("CPP graph", list(WINDOW_OPTIONS.keys()), index=0)  # Select CPP graph window
prx_window_label = st.sidebar.selectbox("PRx graph", list(WINDOW_OPTIONS.keys()), index=0)  # Select PRx graph window
cppopt_window_label = st.sidebar.selectbox("CPPopt graph", list(WINDOW_OPTIONS.keys()), index=0)  # Select CPPopt graph window
cpp_window_minutes = WINDOW_OPTIONS[cpp_window_label]  # Convert CPP graph window to minutes
prx_window_minutes = WINDOW_OPTIONS[prx_window_label]  # Convert PRx graph window to minutes
cppopt_window_minutes = WINDOW_OPTIONS[cppopt_window_label]  # Convert CPPopt graph window to minutes

# =========================================================
# LIVE CSV PROCESSING
# =========================================================

try:  # Start live CSV reading
    df = pd.read_csv(DATA_FILE)  # Read the current live CSV file

    if st.session_state.last_processed_row > len(df):  # Check whether writer restarted and CSV became shorter
        for key, value in state_defaults.items():  # Reset session-state values after file restart
            st.session_state[key] = value.copy() if isinstance(value, list) else value  # Reset each key safely

    new_rows = df.iloc[st.session_state.last_processed_row:]  # Select only newly added CSV rows

    for _, row in new_rows.iterrows():  # Process each new monitor sample one by one
        current_timestamp = parse_timestamp(row)  # Convert TTDate and TTTime into monitor datetime
        st.session_state.latest_monitor_timestamp = current_timestamp  # Save latest monitor timestamp for display

        gap_found, gap_sec = check_gap(  # Check whether current row has a timestamp gap from previous row
            st.session_state.previous_timestamp,  # Use previous monitor timestamp
            current_timestamp  # Use current monitor timestamp
        )

        if gap_found:  # Handle a detected monitor data gap
            insert_cpp_nan_gap_points(st.session_state.previous_timestamp, current_timestamp)  # Insert NaN points into CPP graph
            insert_prx_cppopt_nan_gap_points(st.session_state.previous_timestamp, current_timestamp)  # Insert NaN points into PRx and CPPopt graphs
            reset_continuous_windows_after_gap()  # Reset rolling clinical calculations after gap
            st.session_state.latest_gap_message = (  # Save readable gap warning
                f"Data gap detected: {gap_sec:.0f} seconds from "
                f"{st.session_state.previous_timestamp.strftime('%H:%M:%S')} to "
                f"{current_timestamp.strftime('%H:%M:%S')}."
            )  # Create visible gap warning
        else:  # Handle normal continuous data
            st.session_state.latest_gap_message = ""  # Remove old gap warning when continuous data arrives

        map_value = pd.to_numeric(row["mean1"], errors="coerce")  # Read MAP from mean1 and convert invalid values to NaN
        icp_value = pd.to_numeric(row["mean2"], errors="coerce")  # Read ICP from mean2 and convert invalid values to NaN

        if np.isnan(map_value) or np.isnan(icp_value):  # Check whether MAP or ICP is missing
            st.session_state.previous_map = st.session_state.latest_map  # Save previous MAP for trend
            st.session_state.previous_icp = st.session_state.latest_icp  # Save previous ICP for trend
            st.session_state.previous_cpp = st.session_state.latest_cpp  # Save previous CPP for trend
            st.session_state.latest_map = np.nan  # Mark MAP unavailable
            st.session_state.latest_icp = np.nan  # Mark ICP unavailable
            st.session_state.latest_cpp = np.nan  # Mark CPP unavailable
            st.session_state.latest_prx = np.nan  # Mark PRx unavailable
            st.session_state.latest_prx_status = "MAP or ICP sample is missing"  # Explain PRx unavailability
            st.session_state.latest_autoregulation_status = "Not available"  # Mark autoregulation unavailable
            st.session_state.cpp_plot_time.append(current_timestamp)  # Add timestamp to CPP graph
            st.session_state.cpp_plot_values.append(np.nan)  # Add NaN to break CPP graph
            st.session_state.prx_plot_time.append(current_timestamp)  # Add timestamp to PRx graph
            st.session_state.prx_plot_values.append(np.nan)  # Add NaN to break PRx graph
            st.session_state.cppopt_plot_time.append(current_timestamp)  # Add timestamp to CPPopt graph
            st.session_state.cppopt_plot_values.append(np.nan)  # Add NaN to break CPPopt graph
            reset_continuous_windows_after_gap()  # Reset rolling calculations because sample is incomplete
            st.session_state.previous_timestamp = current_timestamp  # Save timestamp for next gap comparison
            continue  # Move to next CSV row

        cpp_value = calculate_cpp(map_value, icp_value)  # Calculate CPP as MAP minus ICP

        st.session_state.previous_map = st.session_state.latest_map  # Save previous MAP before updating
        st.session_state.previous_icp = st.session_state.latest_icp  # Save previous ICP before updating
        st.session_state.previous_cpp = st.session_state.latest_cpp  # Save previous CPP before updating
        st.session_state.latest_map = map_value  # Update live MAP value
        st.session_state.latest_icp = icp_value  # Update live ICP value
        st.session_state.latest_cpp = cpp_value  # Update live CPP value

        st.session_state.map_history.append(map_value)  # Add MAP to continuous PRx history
        st.session_state.icp_history.append(icp_value)  # Add ICP to continuous PRx history
        st.session_state.cpp_history.append(cpp_value)  # Add CPP to continuous mean CPP history
        st.session_state.cpp_plot_time.append(current_timestamp)  # Add actual monitor time to CPP graph
        st.session_state.cpp_plot_values.append(cpp_value)  # Add CPP value to CPP graph
        st.session_state.previous_timestamp = current_timestamp  # Save current timestamp for next gap detection

        enough_prx_samples = len(st.session_state.map_history) >= PRX_WINDOW_SAMPLES  # Check whether 60 continuous samples are available
        first_prx = st.session_state.last_prx_calculation_time is None  # Check whether no PRx calculation has occurred yet

        if first_prx:  # Handle first PRx calculation
            prx_due = enough_prx_samples  # Calculate first PRx exactly when the 60th continuous sample arrives
        else:  # Handle later PRx calculations
            elapsed_seconds = (current_timestamp - st.session_state.last_prx_calculation_time).total_seconds()  # Calculate monitor seconds since previous PRx attempt
            prx_due = enough_prx_samples and elapsed_seconds >= PRX_UPDATE_INTERVAL_SEC  # Calculate later PRx every 60 monitor seconds

        if prx_due:  # Calculate PRx and CPPopt only when clinically due
            map_window = st.session_state.map_history[-PRX_WINDOW_SAMPLES:]  # Select latest 60 MAP samples
            icp_window = st.session_state.icp_history[-PRX_WINDOW_SAMPLES:]  # Select latest 60 ICP samples
            cpp_window = st.session_state.cpp_history[-PRX_WINDOW_SAMPLES:]  # Select latest 60 CPP samples

            prx_value, prx_status = calculate_prx(map_window, icp_window)  # Calculate 5-minute rolling PRx
            mean_cpp_value = calculate_mean_cpp(cpp_window)  # Calculate mean CPP from same 60 samples

            st.session_state.previous_prx = st.session_state.latest_prx  # Save previous PRx before updating
            st.session_state.latest_prx = prx_value  # Update live PRx value
            st.session_state.latest_prx_status = prx_status if not np.isnan(prx_value) else (prx_status if prx_status != "Valid" else "PRx value is unavailable")  # Store safe PRx status
            st.session_state.prx_plot_time.append(current_timestamp)  # Add PRx timestamp to graph
            st.session_state.prx_plot_values.append(prx_value)  # Add PRx value or NaN to graph
            st.session_state.last_prx_calculation_time = current_timestamp  # Save exact PRx attempt time

            if np.isnan(prx_value):  # Check whether PRx is unavailable
                st.session_state.latest_autoregulation_status = "Not available"  # Mark autoregulation unavailable
                st.session_state.latest_cppopt = np.nan  # Mark current CPPopt unavailable
                st.session_state.latest_cppopt_status = f"PRx unavailable: {st.session_state.latest_prx_status}"  # Explain CPPopt unavailability
                st.session_state.cppopt_plot_time.append(current_timestamp)  # Add timestamp to CPPopt graph
                st.session_state.cppopt_plot_values.append(np.nan)  # Add NaN to CPPopt graph
            else:  # Handle valid PRx only
                st.session_state.prx_history.append(prx_value)  # Store only valid PRx values for CPPopt
                st.session_state.mean_cpp_history.append(mean_cpp_value)  # Store paired mean CPP only with valid PRx

                if prx_value < PRX_AUTOREGULATION_THRESHOLD:  # Check PRx against autoregulation threshold
                    st.session_state.latest_autoregulation_status = "Preserved autoregulation"  # Mark autoregulation preserved
                else:  # Handle PRx values equal to or above threshold
                    st.session_state.latest_autoregulation_status = "Impaired autoregulation"  # Mark autoregulation impaired

                cppopt_value, cppopt_status = calculate_cppopt(  # Calculate CPPopt from valid PRx-mean CPP pairs only
                    np.asarray(st.session_state.prx_history, dtype=float),  # Convert valid PRx history to numeric array
                    np.asarray(st.session_state.mean_cpp_history, dtype=float)  # Convert paired mean CPP history to numeric array
                )

                st.session_state.previous_cppopt = st.session_state.latest_cppopt  # Save previous CPPopt before updating
                st.session_state.latest_cppopt = cppopt_value  # Update live CPPopt value
                st.session_state.latest_cppopt_status = cppopt_status  # Save CPPopt availability reason
                st.session_state.cppopt_plot_time.append(current_timestamp)  # Add timestamp to CPPopt graph
                st.session_state.cppopt_plot_values.append(cppopt_value)  # Add CPPopt value or NaN to graph

                if not np.isnan(cppopt_value):  # Check whether CPPopt is valid
                    st.session_state.last_valid_cppopt = cppopt_value  # Save last valid CPPopt value
                    st.session_state.last_valid_cppopt_time = current_timestamp  # Save last valid CPPopt timestamp

    st.session_state.last_processed_row = len(df)  # Mark all currently read rows as processed
    trim_history()  # Limit graph histories for stable performance

except FileNotFoundError:  # Handle missing live CSV file
    st.error("live_data.csv was not found in the same folder as dashboard_3.py")  # Display clear file error

except Exception as error:  # Handle all other dashboard errors
    st.error(f"Dashboard error: {error}")  # Display exact error message

# =========================================================
# DERIVED DISPLAY VALUES
# =========================================================

if np.isnan(st.session_state.latest_cppopt) or np.isnan(st.session_state.latest_cpp):  # Check whether CPPopt minus CPP can be calculated
    cppopt_minus_cpp = np.nan  # Mark difference unavailable
else:  # Handle valid CPPopt and CPP
    cppopt_minus_cpp = st.session_state.latest_cppopt - st.session_state.latest_cpp  # Calculate CPPopt minus CPP

if not np.isnan(cppopt_minus_cpp):  # Check whether current difference is available
    st.session_state.previous_cppopt_minus_cpp = st.session_state.previous_cppopt_minus_cpp  # Keep previous difference unchanged unless later feature uses it

# =========================================================
# HEADER AND TOP LIVE CARDS
# =========================================================

# The patient information panel was intentionally removed to keep the ICU display focused on live physiology.
st.markdown('<div class="main-title">Real-Time Cerebral Monitoring Dashboard</div>', unsafe_allow_html=True)  # Display main title with theme-safe color
st.markdown('<div class="subtitle">CPP, PRx and CPPopt monitoring using live CSV data</div>', unsafe_allow_html=True)  # Display subtitle

st.divider()  # Draw visual separator

map_col, icp_col, last_cppopt_col, time_col = st.columns([1, 1, 1, 2])  # Create top row for MAP, ICP, last valid CPPopt and monitor timestamp

with map_col:  # Render MAP card
    render_metric_card(
        "MAP (mmHg)",  # Card label
        format_live_value(st.session_state.latest_map, 2),  # Card value
        MAP_COLOR,  # Card color
        trend_arrow(st.session_state.latest_map, st.session_state.previous_map),  # Trend arrow
        "Mean arterial pressure"  # Subtext
    )

with icp_col:  # Render ICP card
    render_metric_card(
        "ICP (mmHg)",  # Card label
        format_live_value(st.session_state.latest_icp, 2),  # Card value
        ICP_COLOR,  # Card color
        trend_arrow(st.session_state.latest_icp, st.session_state.previous_icp),  # Trend arrow
        "Intracranial pressure"  # Subtext
    )

with last_cppopt_col:  # Render last valid CPPopt card immediately after ICP
    if np.isnan(st.session_state.last_valid_cppopt):  # Check whether any valid CPPopt exists
        last_cppopt_value_text = "Not available"  # Show unavailable state
        last_cppopt_subtext = "Last valid CPPopt"  # Explain card purpose
    else:  # Handle available last valid CPPopt
        last_cppopt_value_text = format_live_value(st.session_state.last_valid_cppopt, 2)  # Format last valid CPPopt
        if st.session_state.last_valid_cppopt_time is None:  # Check whether timestamp is missing
            last_cppopt_subtext = "Last valid CPPopt"  # Use simple subtext
        else:  # Handle available timestamp
            last_cppopt_subtext = st.session_state.last_valid_cppopt_time.strftime("%H:%M:%S")  # Show time of last valid CPPopt
    render_metric_card("Last CPPopt", last_cppopt_value_text, CPPOPT_COLOR, "", last_cppopt_subtext)  # Render last CPPopt card

with time_col:  # Render monitor timestamp card
    if st.session_state.latest_monitor_timestamp is None:  # Check whether monitor timestamp exists
        monitor_time_text = "Waiting"  # Display waiting state
        monitor_subtext = "No monitor timestamp received"  # Display timestamp subtext
    else:  # Handle available monitor timestamp
        monitor_time_text = st.session_state.latest_monitor_timestamp.strftime("%H:%M:%S")  # Display TTTime as main value
        monitor_subtext = st.session_state.latest_monitor_timestamp.strftime("%d-%m-%Y")  # Display TTDate as subtext
    render_metric_card("Current Time", monitor_time_text, "#334155", "", monitor_subtext)  # Render timestamp card

if st.session_state.latest_gap_message:  # Check whether a timestamp gap was found
    st.warning(st.session_state.latest_gap_message)  # Display data-gap warning

if np.isnan(st.session_state.latest_prx):  # Check whether current PRx is unavailable
    st.info(f"PRx unavailable: {st.session_state.latest_prx_status}")  # Display exact PRx reason

if np.isnan(st.session_state.latest_cppopt):  # Check whether current CPPopt is unavailable
    st.info(f"CPPopt unavailable: {st.session_state.latest_cppopt_status}")  # Display exact CPPopt reason

# =========================================================
# GRAPH SECTION 1: CPP
# =========================================================

st.subheader("CPP vs Time")  # Display CPP graph heading
cpp_graph_col, cpp_value_col = st.columns([4, 1])  # Create graph-left and live-value-right layout

with cpp_graph_col:  # Render CPP graph area
    cpp_plot = make_plot_dataframe(st.session_state.cpp_plot_time, st.session_state.cpp_plot_values, "CPP", cpp_window_minutes)  # Build cropped CPP graph dataframe
    render_line_chart(cpp_plot, "CPP", CPP_COLOR, "CPP")  # Render CPP graph

with cpp_value_col:  # Render CPP live value beside graph
    render_side_value(
        "CPP",  # Side value title
        format_live_value(st.session_state.latest_cpp, 2),  # Latest CPP value
        CPP_COLOR,  # CPP color
        trend_arrow(st.session_state.latest_cpp, st.session_state.previous_cpp),  # CPP trend arrow
        f"Window: {cpp_window_label}"  # Display selected graph window
    )

# =========================================================
# GRAPH SECTION 2: PRx + AUTOREGULATION
# =========================================================

st.subheader("PRx vs Time")  # Display PRx graph heading
prx_graph_col, prx_value_col = st.columns([4, 1])  # Create graph-left and live-value-right layout

with prx_graph_col:  # Render PRx graph area
    prx_plot = make_plot_dataframe(st.session_state.prx_plot_time, st.session_state.prx_plot_values, "PRx", prx_window_minutes)  # Build cropped PRx graph dataframe
    render_line_chart(prx_plot, "PRx", PRX_COLOR, "PRx")  # Render PRx graph

with prx_value_col:  # Render PRx live value beside graph
    render_side_value(
        "PRx",  # Side value title
        format_live_value(st.session_state.latest_prx, 3),  # Latest PRx value
        PRX_COLOR,  # PRx color
        trend_arrow(st.session_state.latest_prx, st.session_state.previous_prx),  # PRx trend arrow
        f"Window: {prx_window_label}"  # Display selected graph window
    )
    render_autoregulation_badge(st.session_state.latest_autoregulation_status)  # Render color-coded autoregulation badge

# =========================================================
# GRAPH SECTION 3: CPPopt + CPPopt - CPP
# =========================================================

st.subheader("CPPopt vs Time")  # Display CPPopt graph heading
cppopt_graph_col, cppopt_value_col = st.columns([4, 1])  # Create graph-left and live-value-right layout

with cppopt_graph_col:  # Render CPPopt graph area
    cppopt_plot = make_plot_dataframe(st.session_state.cppopt_plot_time, st.session_state.cppopt_plot_values, "CPPopt", cppopt_window_minutes)  # Build cropped CPPopt graph dataframe
    render_line_chart(cppopt_plot, "CPPopt", CPPOPT_COLOR, "CPPopt")  # Render CPPopt graph

with cppopt_value_col:  # Render CPPopt live value beside graph
    render_side_value(
        "CPPopt",  # Side value title
        format_live_value(st.session_state.latest_cppopt, 2),  # Latest CPPopt value
        CPPOPT_COLOR,  # CPPopt color
        trend_arrow(st.session_state.latest_cppopt, st.session_state.previous_cppopt),  # CPPopt trend arrow
        f"Window: {cppopt_window_label}"  # Display selected graph window
    )
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)  # Add spacing between value cards
    render_side_value(
        "CPPopt − CPP",  # Side value title
        format_live_value(cppopt_minus_cpp, 2),  # Latest CPPopt minus CPP value
        DIFF_COLOR,  # Difference color
        trend_arrow(cppopt_minus_cpp, st.session_state.previous_cppopt_minus_cpp),  # Difference trend arrow
        "Target difference"  # Side caption
    )

# =========================================================
# REFRESH
# =========================================================

time.sleep(DASHBOARD_REFRESH_SEC)  # Wait one second before checking the CSV again
st.rerun()  # Refresh the dashboard in a controlled manner
