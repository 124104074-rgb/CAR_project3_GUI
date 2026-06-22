import streamlit as st
import pandas as pd
import numpy as np

from analysis import (
    calculate_cpp,
    calculate_prx,
    calculate_cppopt,
    PRX_WINDOW_SAMPLES
)

st.set_page_config(
    page_title="CPPopt Monitor",
    layout="wide"
)

DATA_FILE = "live_data.csv"

# ==================================================
# SESSION STATE
# ==================================================

if "last_processed_row" not in st.session_state:
    st.session_state.last_processed_row = 0

if "map_history" not in st.session_state:
    st.session_state.map_history = []

if "icp_history" not in st.session_state:
    st.session_state.icp_history = []

if "cpp_history" not in st.session_state:
    st.session_state.cpp_history = []

if "cpp_time" not in st.session_state:
    st.session_state.cpp_time = []

if "prx_history" not in st.session_state:
    st.session_state.prx_history = []

if "prx_time" not in st.session_state:
    st.session_state.prx_time = []

if "mean_cpp_history" not in st.session_state:
    st.session_state.mean_cpp_history = []

if "cppopt_history" not in st.session_state:
    st.session_state.cppopt_history = []

if "cppopt_time" not in st.session_state:
    st.session_state.cppopt_time = []

if "sample_counter" not in st.session_state:
    st.session_state.sample_counter = 0

if "latest_prx" not in st.session_state:
    st.session_state.latest_prx = np.nan

if "latest_cppopt" not in st.session_state:
    st.session_state.latest_cppopt = np.nan

# ==================================================
# READ DATA
# ==================================================

try:
    df = pd.read_csv(DATA_FILE)

    new_rows = df.iloc[st.session_state.last_processed_row:]

    for _, row in new_rows.iterrows():
        map_value = float(row["mean1"])
        icp_value = float(row["mean2"])

        cpp_value = calculate_cpp(map_value, icp_value)

        st.session_state.map_history.append(map_value)
        st.session_state.icp_history.append(icp_value)
        st.session_state.cpp_history.append(cpp_value)

        st.session_state.sample_counter += 1

        current_time = st.session_state.sample_counter * 5
        st.session_state.cpp_time.append(current_time)

        # ==========================================
        # PRx every 12 samples
        # ==========================================
        if (
            len(st.session_state.map_history) >= PRX_WINDOW_SAMPLES
            and st.session_state.sample_counter % 12 == 0
        ):
            map_window = st.session_state.map_history[-PRX_WINDOW_SAMPLES:]
            icp_window = st.session_state.icp_history[-PRX_WINDOW_SAMPLES:]
            cpp_window = st.session_state.cpp_history[-PRX_WINDOW_SAMPLES:]

            prx, _ = calculate_prx(map_window, icp_window)
            mean_cpp = np.mean(cpp_window)

            st.session_state.latest_prx = prx
            st.session_state.prx_history.append(prx)
            st.session_state.prx_time.append(current_time)

            st.session_state.mean_cpp_history.append(mean_cpp)

            cppopt, _ = calculate_cppopt(
                st.session_state.prx_history,
                st.session_state.mean_cpp_history
            )

            st.session_state.latest_cppopt = cppopt
            st.session_state.cppopt_history.append(cppopt)
            st.session_state.cppopt_time.append(current_time)

    st.session_state.last_processed_row = len(df)

except Exception as e:
    st.error(str(e))

# ==================================================
# LATEST VALUES
# ==================================================

if len(st.session_state.map_history) > 0:
    latest_map = st.session_state.map_history[-1]
    latest_icp = st.session_state.icp_history[-1]
    latest_cpp = st.session_state.cpp_history[-1]

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("MAP", f"{latest_map:.2f}")
    c2.metric("ICP", f"{latest_icp:.2f}")
    c3.metric("CPP", f"{latest_cpp:.2f}")
    c4.metric("PRx", f"{st.session_state.latest_prx:.3f}")
    c5.metric("CPPopt", f"{st.session_state.latest_cppopt:.2f}")

# ==================================================
# CPP GRAPH
# ==================================================

st.subheader("CPP vs Time")

cpp_df = pd.DataFrame({
    "Time": st.session_state.cpp_time,
    "CPP": st.session_state.cpp_history
})

st.line_chart(cpp_df.set_index("Time"))

# ==================================================
# PRx GRAPH
# ==================================================

st.subheader("PRx vs Time")

if len(st.session_state.prx_history) > 0:
    prx_df = pd.DataFrame({
        "Time": st.session_state.prx_time,
        "PRx": st.session_state.prx_history
    })

    st.line_chart(prx_df.set_index("Time"))

# ==================================================
# CPPopt GRAPH
# ==================================================

st.subheader("CPPopt vs Time")

if len(st.session_state.cppopt_history) > 0:
    cppopt_df = pd.DataFrame({
        "Time": st.session_state.cppopt_time,
        "CPPopt": st.session_state.cppopt_history
    })

    st.line_chart(cppopt_df.set_index("Time"))

st.rerun()