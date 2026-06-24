import streamlit as st  # Import Streamlit for dashboard interface
import pandas as pd  # Import pandas for CSV reading and graph tables
import numpy as np  # Import NumPy for NaN handling and numerical operations
import time  # Import time for controlled dashboard refresh
from datetime import datetime, timedelta  # Import datetime tools for monitor timestamps and gap points

from analysis_1 import (  # Import analysis functions and constants from analysis.py
    calculate_cpp,  # Import CPP calculation function
    calculate_prx,  # Import PRx calculation function
    calculate_mean_cpp,  # Import mean CPP calculation function
    calculate_cppopt,  # Import CPPopt calculation function
    check_gap,  # Import timestamp gap detection function
    PRX_WINDOW_SAMPLES,  # Import PRx rolling-window sample count
    REAL_SAMPLE_INTERVAL  # Import expected real monitor sample interval
)

st.set_page_config(page_title="CPPopt Monitor", layout="wide")  # Configure Streamlit browser page
st.title("Real-Time Cerebral Monitoring Dashboard")  # Display dashboard title

DATA_FILE = "live_data.csv"  # Define the incoming live CSV file name
PRX_UPDATE_INTERVAL_SEC = 60  # Define PRx and CPPopt update interval in real monitor seconds
PRX_AUTOREGULATION_THRESHOLD = 0.25  # Define PRx threshold for autoregulation classification
DASHBOARD_REFRESH_SEC = 1.0  # Refresh dashboard once per second
MAX_CPP_GRAPH_POINTS = 5000  # Limit CPP graph history to prevent performance problems
MAX_PRX_GRAPH_POINTS = 1000  # Limit PRx graph history to prevent performance problems
MAX_CPPOPT_GRAPH_POINTS = 1000  # Limit CPPopt graph history to prevent performance problems


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


def format_live_value(value, decimals=2):  # Convert a numeric value into dashboard display text
    if value is None or np.isnan(value):  # Check whether the value is unavailable
        return "Not available"  # Display unavailable text
    return f"{value:.{decimals}f}"  # Display formatted numeric value


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


if "last_processed_row" not in st.session_state:  # Create CSV processed-row tracker once
    st.session_state.last_processed_row = 0  # Start processing from the first CSV row

if "map_history" not in st.session_state:  # Create MAP rolling history once
    st.session_state.map_history = []  # Store continuous MAP samples for PRx

if "icp_history" not in st.session_state:  # Create ICP rolling history once
    st.session_state.icp_history = []  # Store continuous ICP samples for PRx

if "cpp_history" not in st.session_state:  # Create CPP rolling history once
    st.session_state.cpp_history = []  # Store continuous CPP samples for mean CPP

if "prx_history" not in st.session_state:  # Create valid PRx history once
    st.session_state.prx_history = []  # Store only valid PRx values for CPPopt

if "mean_cpp_history" not in st.session_state:  # Create paired mean CPP history once
    st.session_state.mean_cpp_history = []  # Store mean CPP values paired only with valid PRx values

if "cpp_plot_time" not in st.session_state:  # Create CPP graph timestamps once
    st.session_state.cpp_plot_time = []  # Store CPP graph timestamps

if "cpp_plot_values" not in st.session_state:  # Create CPP graph values once
    st.session_state.cpp_plot_values = []  # Store CPP values and NaN gap points

if "prx_plot_time" not in st.session_state:  # Create PRx graph timestamps once
    st.session_state.prx_plot_time = []  # Store PRx graph timestamps

if "prx_plot_values" not in st.session_state:  # Create PRx graph values once
    st.session_state.prx_plot_values = []  # Store PRx values and NaN gap points

if "cppopt_plot_time" not in st.session_state:  # Create CPPopt graph timestamps once
    st.session_state.cppopt_plot_time = []  # Store CPPopt graph timestamps

if "cppopt_plot_values" not in st.session_state:  # Create CPPopt graph values once
    st.session_state.cppopt_plot_values = []  # Store CPPopt values and NaN gap points

if "previous_timestamp" not in st.session_state:  # Create previous monitor timestamp storage once
    st.session_state.previous_timestamp = None  # No previous timestamp exists initially

if "last_prx_calculation_time" not in st.session_state:  # Create PRx timing tracker once
    st.session_state.last_prx_calculation_time = None  # No PRx calculation has happened initially

if "latest_map" not in st.session_state:  # Create latest MAP storage once
    st.session_state.latest_map = np.nan  # Start MAP as unavailable

if "latest_icp" not in st.session_state:  # Create latest ICP storage once
    st.session_state.latest_icp = np.nan  # Start ICP as unavailable

if "latest_cpp" not in st.session_state:  # Create latest CPP storage once
    st.session_state.latest_cpp = np.nan  # Start CPP as unavailable

if "latest_prx" not in st.session_state:  # Create latest PRx storage once
    st.session_state.latest_prx = np.nan  # Start PRx as unavailable

if "latest_prx_status" not in st.session_state:  # Create PRx status storage once
    st.session_state.latest_prx_status = "Waiting for 60 continuous samples"  # Explain initial PRx state

if "latest_cppopt" not in st.session_state:  # Create latest CPPopt storage once
    st.session_state.latest_cppopt = np.nan  # Start CPPopt as unavailable

if "latest_cppopt_status" not in st.session_state:  # Create CPPopt status storage once
    st.session_state.latest_cppopt_status = "Waiting for valid PRx values"  # Explain initial CPPopt state

if "last_valid_cppopt" not in st.session_state:  # Create last valid CPPopt storage once
    st.session_state.last_valid_cppopt = np.nan  # Start with no previous valid CPPopt

if "last_valid_cppopt_time" not in st.session_state:  # Create last valid CPPopt timestamp storage once
    st.session_state.last_valid_cppopt_time = None  # Start with no valid CPPopt time

if "latest_autoregulation_status" not in st.session_state:  # Create autoregulation status storage once
    st.session_state.latest_autoregulation_status = "Not available"  # Start autoregulation as unavailable

if "latest_gap_message" not in st.session_state:  # Create gap warning storage once
    st.session_state.latest_gap_message = ""  # Start with no gap warning

if "latest_monitor_timestamp" not in st.session_state:  # Create monitor timestamp display storage once
    st.session_state.latest_monitor_timestamp = None  # Start with no monitor timestamp


try:  # Start live CSV reading
    df = pd.read_csv(DATA_FILE)  # Read the current live CSV file

    if st.session_state.last_processed_row > len(df):  # Check whether writer restarted and CSV became shorter
        st.session_state.last_processed_row = 0  # Restart dashboard processing from first row
        st.session_state.map_history.clear()  # Clear old MAP history
        st.session_state.icp_history.clear()  # Clear old ICP history
        st.session_state.cpp_history.clear()  # Clear old CPP history
        st.session_state.prx_history.clear()  # Clear old PRx history
        st.session_state.mean_cpp_history.clear()  # Clear old mean CPP history
        st.session_state.cpp_plot_time.clear()  # Clear old CPP graph timestamps
        st.session_state.cpp_plot_values.clear()  # Clear old CPP graph values
        st.session_state.prx_plot_time.clear()  # Clear old PRx graph timestamps
        st.session_state.prx_plot_values.clear()  # Clear old PRx graph values
        st.session_state.cppopt_plot_time.clear()  # Clear old CPPopt graph timestamps
        st.session_state.cppopt_plot_values.clear()  # Clear old CPPopt graph values
        st.session_state.previous_timestamp = None  # Reset previous timestamp
        st.session_state.last_prx_calculation_time = None  # Reset PRx timer

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

            st.session_state.latest_prx = prx_value  # Update live PRx value
            st.session_state.latest_prx_status = prx_status  # Save exact PRx calculation status
            st.session_state.prx_plot_time.append(current_timestamp)  # Add PRx timestamp to graph
            st.session_state.prx_plot_values.append(prx_value)  # Add PRx value or NaN to graph
            st.session_state.last_prx_calculation_time = current_timestamp  # Save exact PRx attempt time

            if np.isnan(prx_value):  # Check whether PRx is unavailable
                st.session_state.latest_autoregulation_status = "Not available"  # Mark autoregulation unavailable
                st.session_state.latest_cppopt = np.nan  # Mark current CPPopt unavailable
                st.session_state.latest_cppopt_status = f"PRx unavailable: {prx_status}"  # Explain CPPopt unavailability
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
    st.error("live_data.csv was not found in the same folder as dashboard.py")  # Display clear file error

except Exception as error:  # Handle all other dashboard errors
    st.error(f"Dashboard error: {error}")  # Display exact error message


st.divider()  # Draw visual separation before live cards

col1, col2, col3, col4, col5, col6 = st.columns(6)  # Create six dashboard value blocks

col1.metric("MAP (mmHg)", format_live_value(st.session_state.latest_map, 2))  # Display latest MAP
col2.metric("ICP (mmHg)", format_live_value(st.session_state.latest_icp, 2))  # Display latest ICP
col3.metric("CPP (mmHg)", format_live_value(st.session_state.latest_cpp, 2))  # Display latest CPP
col4.metric("PRx", format_live_value(st.session_state.latest_prx, 3))  # Display latest PRx
col5.metric("CPPopt (mmHg)", format_live_value(st.session_state.latest_cppopt, 2))  # Display latest CPPopt

if np.isnan(st.session_state.latest_cppopt) or np.isnan(st.session_state.latest_cpp):  # Check whether CPPopt minus CPP can be calculated
    cppopt_difference_text = "Not available"  # Mark difference unavailable
else:  # Handle valid CPPopt and CPP
    cppopt_difference_text = f"{st.session_state.latest_cppopt - st.session_state.latest_cpp:.2f}"  # Calculate CPPopt minus CPP

col6.metric("CPPopt − CPP (mmHg)", cppopt_difference_text)  # Display CPPopt difference for clinical review


st.subheader("Latest Monitor Timestamp")  # Display monitor timestamp heading

if st.session_state.latest_monitor_timestamp is None:  # Check whether any monitor row has arrived
    st.info("Waiting for monitor timestamp...")  # Show waiting message
else:  # Handle available monitor timestamp
    st.info(st.session_state.latest_monitor_timestamp.strftime("%d-%m-%Y %H:%M:%S"))  # Display exact TTDate and TTTime


st.subheader("Autoregulation Status")  # Display autoregulation heading
st.write(st.session_state.latest_autoregulation_status)  # Display preserved, impaired, or unavailable status


if st.session_state.latest_gap_message:  # Check whether a timestamp gap was found
    st.warning(st.session_state.latest_gap_message)  # Display data-gap warning


if np.isnan(st.session_state.latest_prx):  # Check whether current PRx is unavailable
    st.info(f"PRx unavailable: {st.session_state.latest_prx_status}")  # Display exact PRx reason


if np.isnan(st.session_state.latest_cppopt):  # Check whether current CPPopt is unavailable
    st.info(f"CPPopt unavailable: {st.session_state.latest_cppopt_status}")  # Display exact CPPopt reason


st.subheader("Last Valid CPPopt")  # Display last valid CPPopt heading

if np.isnan(st.session_state.last_valid_cppopt):  # Check whether a valid CPPopt has ever been calculated
    st.write("No valid CPPopt calculated yet")  # Display no valid CPPopt message
else:  # Handle available previous CPPopt
    last_cppopt_text = f"{st.session_state.last_valid_cppopt:.2f} mmHg"  # Format last valid CPPopt
    last_cppopt_time_text = st.session_state.last_valid_cppopt_time.strftime("%d-%m-%Y %H:%M:%S")  # Format last valid CPPopt time
    st.write(f"{last_cppopt_text} at {last_cppopt_time_text}")  # Display last valid CPPopt and monitor time


st.subheader("CPP vs Time")  # Display CPP graph heading

if len(st.session_state.cpp_plot_time) > 0:  # Plot CPP only when data exists
    cpp_plot = pd.DataFrame({  # Create CPP graph dataframe
        "Time": st.session_state.cpp_plot_time,  # Use monitor timestamps
        "CPP": st.session_state.cpp_plot_values  # Use CPP values and NaN gap points
    })

    cpp_plot["Time"] = pd.to_datetime(cpp_plot["Time"]).dt.strftime("%H:%M:%S")  # Display only TTTime on x-axis
    st.line_chart(cpp_plot.set_index("Time"), use_container_width=True)  # Draw CPP graph with NaN line breaks


st.subheader("PRx vs Time")  # Display PRx graph heading

if len(st.session_state.prx_plot_time) > 0:  # Plot PRx only when data exists
    prx_plot = pd.DataFrame({  # Create PRx graph dataframe
        "Time": st.session_state.prx_plot_time,  # Use PRx monitor timestamps
        "PRx": st.session_state.prx_plot_values  # Use PRx values and NaN gap points
    })

    prx_plot["Time"] = pd.to_datetime(prx_plot["Time"]).dt.strftime("%H:%M:%S")  # Display only TTTime on x-axis
    st.line_chart(prx_plot.set_index("Time"), use_container_width=True)  # Draw PRx graph with NaN line breaks


st.subheader("CPPopt vs Time")  # Display CPPopt graph heading

if len(st.session_state.cppopt_plot_time) > 0:  # Plot CPPopt only when data exists
    cppopt_plot = pd.DataFrame({  # Create CPPopt graph dataframe
        "Time": st.session_state.cppopt_plot_time,  # Use CPPopt monitor timestamps
        "CPPopt": st.session_state.cppopt_plot_values  # Use CPPopt values and NaN gap points
    })

    cppopt_plot["Time"] = pd.to_datetime(cppopt_plot["Time"]).dt.strftime("%H:%M:%S")  # Display only TTTime on x-axis
    st.line_chart(cppopt_plot.set_index("Time"), use_container_width=True)  # Draw CPPopt graph with NaN line breaks


time.sleep(DASHBOARD_REFRESH_SEC)  # Wait one second before checking the CSV again
st.rerun()  # Refresh the dashboard in a controlled manner