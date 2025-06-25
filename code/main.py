import streamlit as st

st.set_page_config(
    page_title="CPEMS Home",
    layout="wide", # Use wide layout
    initial_sidebar_state="expanded" # Keep sidebar open initially
)

# --- Page Configuration ---
# Paths based on the filenames 'pages/1_Forecasting.py' and 'pages/2_Simulation.py'
FORECASTING_PAGE_PATH = "/Forecasting"
SIMULATION_PAGE_PATH = "/Simulation"

# --- Main Page Content ---
st.title("Cyber Physical Energy Management System ⚡")
st.markdown("---") # Separator line

st.subheader("Welcome!")
st.markdown("""
Navigate through the different modules of the CPEMS using the sidebar on the left, or click one of the boxes below.
- **Forecasting:** Train the GRU model and generate multi-horizon energy forecasts.
- **Industrial Profile:** Run MATLAB/Simulink simulations and visualize data from InfluxDB.
""")

st.markdown("---")

# --- Clickable Visual Boxes (using columns and markdown with HTML links) ---
col1, col2 = st.columns(2)

# Common style for the links to make them behave like block elements and remove underlines
link_style = "text-decoration: none; color: inherit; display: block; height: 100%;"

# Common style for the inner div content box
# Added flexbox properties for better vertical centering of content
div_common_style = "border-radius: 10px; padding: 20px; text-align: center; height: 200px; display: flex; flex-direction: column; justify-content: center; align-items: center;"

with col1:
    # Specific styles for the forecasting box
    forecasting_div_style = f"{div_common_style} border: 2px solid #4CAF50; background-color: #f0fff0;"
 st.markdown(
        f"""
        <a href="{SIMULATION_PAGE_PATH}" target="_self" style="{link_style}">
            <div style="{simulation_div_style}">
                <h2 style="color: #2196F3; margin-bottom: 15px;">⚙️ Industrial Profile</h2>
                <p style="color: #333; margin-bottom: 10px;">Run MATLAB simulations, upload data to InfluxDB, and visualize simulation results.</p>
                <p style="color: #555; font-style: italic; font-size: 0.9em;">Click this box to navigate</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

with col2:
    # Specific styles for the simulation box
    simulation_div_style = f"{div_common_style} border: 2px solid #2196F3; background-color: #f0f8ff;"
    st.markdown(
        f"""
        <a href="{SIMULATION_PAGE_PATH}" target="_self" style="{link_style}">
            <div style="{simulation_div_style}">
                <h2 style="color: #2196F3; margin-bottom: 15px;">Industrial Profile 🏭</h2>
                <p style="color: #333; margin-bottom: 10px;">Run MATLAB simulations, upload data to InfluxDB, and visualize simulation results.</p>
                <p style="color: #555; font-style: italic; font-size: 0.9em;">Click this box to navigate</p>
            </div>
        </a>
        """,
        unsafe_allow_html=True
    )

st.markdown("---") # Separator line at the bottom

# Update sidebar message
st.sidebar.success("Select a page above or click a main panel box.")

# Add any other information you want on the main landing page below
# st.info("System Status: OK")
