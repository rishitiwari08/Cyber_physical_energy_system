# --- Imports ---
import streamlit as st
import pandas as pd
import numpy as np
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime, timedelta, time # Import time for time_input
import os # Import os for file path operations
import sys # To check Python version

# --- MATLAB Availability Check ---
try:
    import matlab.engine
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False

# --- Original Logic Classes (Adapted for Integration) ---

class MatlabDataLoader:
    """ Handles MATLAB simulation execution and data writing. """
    def __init__(self, condition, matlab_script_path, influx_url, token, org, bucket, max_iterations=5000):
        self.condition = condition
        self.matlab_script_path = matlab_script_path
        self.bucket = bucket
        self.org = org
        self.token = token # Store token for writing
        self.max_iterations = max_iterations
        self.client = influxdb_client.InfluxDBClient(url=influx_url, token=self.token, org=org, verify_ssl=False)
        self.eng = None
        self.function_output = None
        self.start_time = datetime.utcnow() # Reference time (t=0) for simulation offsets

    def start_matlab_engine(self):
        """ Start the MATLAB engine and change directory."""
        st.info("Starting MATLAB engine ...")
        try:
            self.eng = matlab.engine.start_matlab()
            self.eng.cd(self.matlab_script_path, nargout=0)
            st.success("MATLAB engine started.")
            return True
        except Exception as e:
            st.error(f"Failed to start MATLAB engine or change directory: {e}")
            st.info(f"Ensure MATLAB is running and the path '{self.matlab_script_path}' is correct.")
            self.eng = None
            return False

    def run_simulation(self):
        """ Run the Simulink function via MATLAB engine."""
        if self.eng is None:
            if not self.start_matlab_engine():
                return False # Indicate failure

        model_name = "Switch"
        condition_value = float(self.condition)

        st.info(f"Running MATLAB simulation: Model = '{model_name}', Condition ={condition_value}...")
        try:
            raw_output = self.eng.Run_Simulink_Function(model_name, condition_value, nargout=1)
            self.function_output = np.array(raw_output)

            if self.function_output is None or self.function_output.size == 0:
                st.error("MATLAB function returned no output.")
                self.function_output = None
                return False

            if len(self.function_output) > self.max_iterations:
                st.warning(f"Output truncated to {self.max_iterations} iterations.")
                self.function_output = self.function_output[:self.max_iterations]

            st.success(f"MATLAB simulation done. Output shape: {self.function_output.shape}")
            return True

        except matlab.engine.MatlabExecutionError as me:
            st.error(f"MATLAB Execution Error: {me}")
            st.error("Check the MATLAB script ('Run_Simulink_Function.m') and model ('Switch.slx') for issues.")
            self.function_output = None
            return False
        except Exception as e:
            st.error(f"Error during MATLAB simulation run: {e}")
            self.function_output = None
            return False

    # ------ ------- ------- ------- ------ ------- ------- ------- ------ -
    # CORRECTED TIMESTAMP FUNCTION
    # ------ ------- ------- ------- ------ ------- ------- ------- ------ -
    def matlab_datenum_to_iso(self, simtime):
        """ Convert MATLAB simulation time offset to ISO format timestamp.
        """
        try:
            # Convert simtime (offset in seconds from start) to float
            simtime_float = float(simtime)
            # Calculate the absolute time for this data point
            actual_time = self.start_time + timedelta(seconds=simtime_float)

            # Generate ISO format string.
            # REMOVED timespec='nanoseconds' which caused 'Unknown timespec' error on Python < 3.11
            # 'auto' (default) includes microseconds if they are non-zero.
            iso_timestamp = actual_time.isoformat(timespec='microseconds') # Explicitly request microseconds

            # Ensure the timestamp ends with 'Z' to denote UTC timezone for InfluxDB
            # self.start_time is UTC, so actual_time is also UTC
            if not iso_timestamp.endswith('Z'):
                # Check if timezone info is already present (e.g., +00:00)
                # This shouldn't happen with utcnow() but check just in case.
                if '+' not in iso_timestamp and '-' not in iso_timestamp[10:]: # Avoid matching date hyphen
                    iso_timestamp += 'Z'

            return iso_timestamp
        except Exception as e:
            # Display error in Streamlit UI for debugging
            st.warning(f"Timestamp conversion error for simtime {repr(simtime)}: {e}")
            return None
    # ------ ------- ------- ------- ------ ------- ------- ------- ------ -

    def write_to_influxdb(self, measurement_name, fields_mapping):
        """ Write simulation output data to InfluxDB."""
        if self.function_output is None or len(self.function_output) == 0:
            st.warning(f"No simulation data available to write for {measurement_name}.")
            return False

        try:
            if not self.client.ping():
                st.error(f"InfluxDB connection failed (ping) for writing to {measurement_name}.")
                return False
        except Exception as ping_e:
            st.error(f"InfluxDB connection error during ping: {ping_e}")
            return False

        write_api = self.client.write_api(write_options=SYNCHRONOUS)
        num_rows = len(self.function_output)
        points_to_write = []
        rows_skipped_ts = 0
        rows_skipped_field = 0
        rows_skipped_nan = 0

        st.info(f"Preparing {num_rows} data points for InfluxDB measurement '{measurement_name}'...")

        for i in range(num_rows):
            row_data = self.function_output[i]
            simtime = row_data[0] # First element is time

            # Use the corrected conversion function
            simtime_iso = self.matlab_datenum_to_iso(simtime)
            if simtime_iso is None:
                rows_skipped_ts += 1
                continue # Skip row if timestamp is invalid

            p = influxdb_client.Point(measurement_name).tag("Machine_Type", str(self.condition))
            valid_field_added = False
            for field, index in fields_mapping.items():
                try:
                    value = float(row_data[index])
                    if not np.isnan(value) and not np.isinf(value):
                        p = p.field(field, value)
                        valid_field_added = True
                    else:
                        rows_skipped_nan += 1
                except IndexError:
                    rows_skipped_field += 1
                    if f"field_idx_warn_{field}_{measurement_name}" not in st.session_state:
                        st.warning(f"Measurement '{measurement_name}': Index {index} out of bounds for field '{field}'.")
                        st.session_state[f"field_idx_warn_{field}_{measurement_name}"] = True
                    valid_field_added = False
                    break # Break if an index is out of bounds for current measurement
                except Exception as field_e:
                    st.warning(f"Measurement '{measurement_name}', Row {i}: Error processing field '{field}': {field_e}")

            if valid_field_added:
                # Still tell the client library the precision is nanoseconds,
                # even if the ISO string only shows microseconds. InfluxDB stores in nanoseconds.
                p = p.time(simtime_iso, write_precision='ns')
                points_to_write.append(p)

        if rows_skipped_ts > 0: st.warning(f"Skipped {rows_skipped_ts} rows due to timestamp errors for {measurement_name}.")
        if rows_skipped_field > 0: st.warning(f"Skipped fields/rows due to index errors for {measurement_name}.")
        if rows_skipped_nan > 0: st.warning(f"Skipped {rows_skipped_nan} NaN/Inf field values for {measurement_name}.")

        if not points_to_write:
            st.warning(f"No valid data points generated to write for {measurement_name}.")
            return False

        st.info(f"Writing {len(points_to_write)} points to InfluxDB '{measurement_name}'...")
        try:
            write_api.write(bucket=self.bucket, org=self.org, record=points_to_write)
            st.success(f"Successfully wrote {len(points_to_write)} points to {measurement_name}.")
            return True
        except Exception as e:
            st.error(f"Error writing to InfluxDB {measurement_name}: {e}")
            return False

    def stop_engine(self):
        """ Explicitly stops the MATLAB engine if running. """
        if self.eng:
            try:
                self.eng.quit()
                st.info("MATLAB engine stopped.")
            except Exception as e:
                st.warning(f"Could not stop MATLAB engine cleanly: {e}")
            self.eng = None


# (InfluxDBQuery and DataVisualization classes remain the same as previous version)
class InfluxDBQuery:
    """ Handles querying data from InfluxDB. """
    def __init__(self, url, token, org, bucket):
        self.org = org
        self.bucket = bucket
        self.client = influxdb_client.InfluxDBClient(url=url, token=token, org=org, verify_ssl=False)

    def query_data(self, measurement, start_time_iso, stop_time_iso):
        """ Query data from InfluxDB using Flux (takes ISO strings). """
        query = f"""
        from(bucket: "{self.bucket}")
          |> range(start: {start_time_iso}, stop: {stop_time_iso})
          |> filter(fn: (r) => r._measurement == "{measurement}")
          |> yield(name: "results")
        """
        st.info(f"Querying InfluxDB: Measurement = '{measurement}', Range = '{start_time_iso}' to '{stop_time_iso}'")
        try:
            query_api = self.client.query_api()
            result = query_api.query(query=query, org=self.org)
            # st.success("InfluxDB query successful.") # Less verbose
            return result
        except Exception as e:
            st.error(f"Error executing InfluxDB query: {e}")
            return None

    def process_records(self, query_result):
        """ Process Flux query result into a Pandas DataFrame. """
        if query_result is None: return pd.DataFrame()
        # st.info("Processing query results into DataFrame ...") # Less verbose
        records = []
        try:
            for table in query_result:
                for record in table.records:
                    records.append({
                        "timestamp": record.get_time(),
                        "Machine_Type": record.values.get("Machine_Type", "N/A"),
                        "_field": record.get_field(),
                        "value": record.get_value()
                    })
            if not records:
                # st.warning("Query returned no records.") # Less verbose
                return pd.DataFrame()
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # st.success(f"DataFrame created with {len(df)} records.") # Less verbose
            return df
        except Exception as e:
            st.error(f"Error processing query results: {e}")
            return pd.DataFrame()


class DataVisualization:
    """ Handles plotting of the queried InfluxDB data. """
    def __init__(self, df):
        if not isinstance(df, pd.DataFrame) or df.empty:
            self.df = pd.DataFrame()
            # st.info("No data available for visualization.") # Less verbose
        else:
            self.df = df

    def display_data(self, selected_field):
        """ Displays time series plot """
        if self.df.empty: return
        if selected_field not in self.df['_field'].unique():
            st.warning(f"Field '{selected_field}' not found in the queried data.")
            return
        df_filtered = self.df[self.df['_field'] == selected_field].copy()
        if df_filtered.empty:
            st.info(f"No data points found for field '{selected_field}'.")
            return
        df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
        df_filtered.sort_values('timestamp', inplace=True)
        st.subheader(f"Time Series Plot for {selected_field}")
        st.line_chart(df_filtered.set_index('timestamp')['value'])

    def display_pivot_table(self):
        """ Displays a pivot table view of the data. """
        st.subheader("Pivoted Data View")
        if self.df.empty:
            st.info("No data to display in table.")
            return
        try:
            df_pivoted = self.df.pivot_table(
                index=["timestamp", "Machine_Type"], columns="_field",
                values="value", aggfunc="first"
            )
            st.dataframe(df_pivoted)
        except Exception as e:
            st.error(f"Could not create pivot table: {e}")


# --- Streamlit App Structure ---
st.set_page_config(page_title="Simulation & Viz", layout="wide")
st.title("Load Selection & Visualization of Energy Consumption 🏭")

# --- Globals / Configuration ---
DEFAULT_MATLAB_SCRIPT_PATH = '/Users/rishikeshtiwari/THK/Case Studies/Case Study IOT' # <--- UPDATE THIS PATH
INFLUX_URL = "https://eu-central-1-1.aws.cloud2.influxdata.com"
# INFLUX_TOKEN_SIM and INFLUX_TOKEN_VIZ should be replaced with your actual InfluxDB tokens
# These tokens are sensitive and should ideally be stored as environment variables or Streamlit secrets, not directly in code.
INFLUX_TOKEN_SIM = "Jzj8SoxpTpDq5FM0JPoYx93wlmJC7BJ7pqYV9GwJMrdp_33DemOXh7SoTIQaLgBfKqnqgdtUAiBilEgd0cksmA=="
INFLUX_TOKEN_VIZ = "ZKu1zQnBhbh1_ywDqH6nQqlVapG_3Nv8v3gA9u4dD_isRGTUtWHprjJwACMO3oG7y2hPkxRRhPvUy23LuUOHVw=="
INFLUX_ORG = "Student"
INFLUX_BUCKET = "CPEMS"

machine_options = {"Motor": 1, "Induction Furnace": 2, "General Electric Load": 4, "Pump": 5}
condition_mapping = {
    (1,): 1, (2,): 2, (1, 2): 3, (4,): 4, (5,): 5, (4, 5): 6, (1, 4): 7,
    (1, 5): 8, (4, 2): 9, (2, 5): 10, (1, 4, 5): 11, (2, 5, 1): 12,
    (4, 2, 1): 13, (2, 4, 5): 14, (1, 2, 4, 5): 15
}
fields_map_siemens = {
    "voltage u1": 1, "voltage u2": 2, "voltage u3": 3, "current i1": 4, "current i2": 5,
    "current i3": 6, "active_power p1": 7, "active_power p2": 8, "active_power p3": 9,
    "reactive_power q1": 10, "reactive_power q2": 11, "reactive_power q3": 12,
    "apparent_power s1": 13, "apparent_power s2": 14, "apparent_power s3": 15,
    "power_factor pf1": 16, "power_factor pf2": 17, "power_factor pf3": 18
}
fields_map_beckhoff = {"voltage u1": 1, "voltage u2": 2, "voltage u3": 3, "current i1": 4, "current i2": 5, "current i3": 6}

# --- Tabs ---
sim_tab, viz_tab = st.tabs(["MATLAB Simulation Control", "Data Visualization"])

# --- Simulation Tab ---
with sim_tab:
    st.header("Run Simulation and Upload")

    # Display Python version for debugging purposes
    st.caption(f"Running on Python {sys.version}")

    if not MATLAB_AVAILABLE:
        st.error(
            "MATLAB Engine for Python not installed or MATLAB not found.\n"
            "Simulation features requiring MATLAB will be disabled.\n\n"
            "**To enable:** Check MATLAB installation and Python engine setup."
        )
    else:
        matlab_script_path_sim = st.text_input(
            "MATLAB Script Directory Path", DEFAULT_MATLAB_SCRIPT_PATH,
            key="sim_path_input"
        )
        selected_machines_sim = st.multiselect(
            "Select Machine Types for Simulation:", list(machine_options.keys()), key="sim_machine_select"
        )
        selected_values_sim = tuple(sorted(machine_options[m] for m in selected_machines_sim))
        condition_sim = condition_mapping.get(selected_values_sim, 0)
        st.write(f"Selected Condition Number: **{condition_sim}**")
        max_iterations_sim = st.number_input(
            "Max Simulation Iterations to Process", min_value=100, value=5000, step=100, key="sim_max_iter"
        )

        if st.button("Run MATLAB Simulation & Upload Data", key="run_sim_button"):
            run_simulation_flag = True
            if not selected_machines_sim:
                st.warning("Please select at least one machine type.")
                run_simulation_flag = False
            if condition_sim == 0 and selected_machines_sim: # If combination not mapped
                st.error("Selected machine combination is not mapped to a valid condition number.")
                run_simulation_flag = False
            if not os.path.isdir(matlab_script_path_sim):
                st.error(f"Invalid MATLAB script directory path: '{matlab_script_path_sim}'.")
                run_simulation_flag = False

            if run_simulation_flag:
                loader = MatlabDataLoader(
                    condition_sim, matlab_script_path_sim, INFLUX_URL,
                    INFLUX_TOKEN_SIM,
                    INFLUX_ORG, INFLUX_BUCKET, max_iterations_sim
                )
                sim_success = loader.run_simulation()
                if sim_success and loader.function_output is not None:
                    st.info("Simulation successful. Proceeding to data upload ...")
                    success1 = loader.write_to_influxdb("Siemens_1", fields_map_siemens)
                    success2 = loader.write_to_influxdb("Beckhoff_1", fields_map_beckhoff)
                    if success1 or success2: st.success("Data upload process completed.")
                    else: st.error("Data upload failed or no valid points written.")
                else:
                    st.error("Simulation failed or produced no data. Cannot upload.")
                loader.stop_engine() # Always stop the engine after attempting simulation


# --- Visualization Tab ---
with viz_tab:
    st.header("Visualize InfluxDB Data")
    influx_querier = InfluxDBQuery(INFLUX_URL, INFLUX_TOKEN_VIZ, INFLUX_ORG, INFLUX_BUCKET)
    viz_query_col1, viz_query_col2 = st.columns(2)
    with viz_query_col1:
        measurements_options = ["Beckhoff_1", "Siemens_1", "Rough_6"] # Add other measurements if needed
        selected_measurement_viz = st.selectbox("Select Measurement", measurements_options, key="viz_measurement_select")
        default_start_datetime = datetime.now() - timedelta(days=1)
        start_date_viz = st.date_input("Start Date", default_start_datetime.date(), key="viz_start_date")
        start_time_viz = st.time_input("Start Time", time(0, 0), key="viz_start_time")
    with viz_query_col2:
        default_end_datetime = datetime.now()
        stop_date_viz = st.date_input("Stop Date", default_end_datetime.date(), key="viz_stop_date")
        stop_time_viz = st.time_input("Stop Time", time(23, 59, 59), key="viz_stop_time")

    start_datetime_viz = datetime.combine(start_date_viz, start_time_viz)
    stop_datetime_viz = datetime.combine(stop_date_viz, stop_time_viz)
    start_time_iso = start_datetime_viz.isoformat(timespec='seconds') + 'Z'
    stop_time_iso = stop_datetime_viz.isoformat(timespec='seconds') + 'Z'

    if st.button("Query InfluxDB Data", key="query_influx_button"):
        query_result = influx_querier.query_data(selected_measurement_viz, start_time_iso, stop_time_iso)
        df_viz = influx_querier.process_records(query_result)
        st.session_state['viz_dataframe'] = df_viz
        st.session_state['viz_measurement'] = selected_measurement_viz
        st.session_state['viz_fields'] = sorted(df_viz['_field'].unique()) if not df_viz.empty else []
        # Clear selection if fields change - prevents errors if old field disappears
        if 'viz_field_select' in st.session_state:
            current_selection = st.session_state['viz_field_select']
            if current_selection not in st.session_state['viz_fields']:
                del st.session_state['viz_field_select']


    if 'viz_dataframe' in st.session_state and isinstance(st.session_state['viz_dataframe'], pd.DataFrame):
        df_display = st.session_state['viz_dataframe']
        if not df_display.empty:
            st.markdown("---")
            st.subheader(f"Visualization for Measurement: {st.session_state.get('viz_measurement', 'N/A')}")
            available_fields = st.session_state.get('viz_fields', [])
            if available_fields:
                # Use session state key directly in selectbox to preserve selection if possible
                selected_field_viz = st.selectbox(
                    "Select Field to Plot", options=available_fields, key="viz_field_select"
                )
                show_pivot_data = st.checkbox('Show Data Table (Pivoted View)', key="viz_show_pivot", value=False)
                try:
                    data_viz = DataVisualization(df_display)
                    data_viz.display_data(selected_field_viz)
                    if show_pivot_data: data_viz.display_pivot_table()
                except Exception as viz_e:
                    st.error(f"An error occurred during visualization: {viz_e}")
            else:
                st.info("Query returned data, but no specific fields were found.")
        elif 'viz_measurement' in st.session_state: # If df_display is empty but a measurement was selected
            st.info(f"No data found for measurement '{st.session_state['viz_measurement']}' in the selected time range.")
