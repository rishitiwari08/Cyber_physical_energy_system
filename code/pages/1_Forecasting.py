# --- Imports for Forecasting ---
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time # Import time
# Note: LabelEncoder is not used, relying on pd.Categorical or numeric days
from torch.utils.data import Dataset, DataLoader
import os
import math
import sys # To check python version if needed

# --- Configuration & Setup ---
st.set_page_config(page_title="GRU Forecasting", layout="wide")
st.title("GRU Load Forecasting 📈")

# Device setup (GPU if available)
@st.cache_resource # Cache the device resource
def get_device():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Forecasting using device: {dev}") # Add print statement for debugging/confirmation
    return dev
device = get_device()
st.sidebar.info(f"Using device: {device}") # Display device info in sidebar


# Paths for model saving and loading
MODEL_PATH = "multi_horizon_gru_model.pth" # Path for the GRU model weights

# Define required input columns for the model and target columns for prediction
# Timestamp is implicitly required by the new load function for resampling
REQUIRED_INPUT_COLUMNS = ["P1", "P2", "P3", "Q1", "Q2", "Q3", "Day"]
TARGET_COLUMNS = ["P1", "P2", "P3", "Q1", "Q2", "Q3"] # Columns the model predicts

# GRU Model Parameters
INPUT_SIZE = len(REQUIRED_INPUT_COLUMNS) # Number of features in input sequence
TARGET_SIZE = len(TARGET_COLUMNS) # Number of features to predict
HIDDEN_SIZE = 128 # Number of features in GRU hidden state
NUM_GRU_LAYERS = 2 # Number of GRU layers
SEQUENCE_LENGTH = 1440 # Input sequence length (e.g., 1 day = 1440 minutes)
DAY_HORIZON = 1440 # Prediction horizon for 1 day
WEEK_HORIZON = 1440 * 7 # Prediction horizon for 7 days


# --- GRU Model Definition ---
class MultiHorizonGRU(nn.Module):
    """
    GRU model predicting multiple horizons (1-day and 7-day) directly.
    """
    def __init__(self, input_size, hidden_size, num_gru_layers, day_horizon, week_horizon, target_size):
        super(MultiHorizonGRU, self).__init__()
        dropout_rate = 0.2 if num_gru_layers > 1 else 0
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_gru_layers,
                          batch_first=True, dropout=dropout_rate)
        # Linear layers to map GRU output to the flattened multi-horizon predictions
        self.fc_day = nn.Linear(hidden_size, day_horizon * target_size)
        self.fc_week = nn.Linear(hidden_size, week_horizon * target_size)
        # Store dimensions for reshaping
        self.day_horizon = day_horizon
        self.week_horizon = week_horizon
        self.target_size = target_size

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        gru_out, h_n = self.gru(x)
        # Use the last hidden state of the last layer for prediction
        x_last = h_n[-1] # shape: (batch, hidden_size)
        # Predict day horizon
        out_day_flat = self.fc_day(x_last) # shape: (batch, day_horizon * target_size)
        out_day = out_day_flat.view(-1, self.day_horizon, self.target_size) # shape: (batch, day_horizon, target_size)
        # Predict week horizon
        out_week_flat = self.fc_week(x_last) # shape: (batch, week_horizon * target_size)
        out_week = out_week_flat.view(-1, self.week_horizon, self.target_size) # shape: (batch, week_horizon, target_size)
        return out_day, out_week

# --- Load or Initialize Model ---
@st.cache_resource # Cache the loaded model object
def load_model(model_path, device):
    """ Loads the GRU model state dict, caching the model instance."""
    model_instance = MultiHorizonGRU(INPUT_SIZE, HIDDEN_SIZE, NUM_GRU_LAYERS, DAY_HORIZON, WEEK_HORIZON, TARGET_SIZE).to(device)
    if os.path.exists(model_path):
        try:
            model_instance.load_state_dict(torch.load(model_path, map_location=device))
            st.sidebar.success("Loaded pre-trained GRU model.")
        except Exception as e:
            st.sidebar.error(f"Error loading GRU state: {e}")
            st.sidebar.warning("Using untrained model.")
    else:
        st.sidebar.warning("No pre-trained GRU model found at specified path.")
    return model_instance

# Load the model once using the cached function
model = load_model(MODEL_PATH, device)

# Define the loss function (used during training)
criterion = nn.SmoothL1Loss()


# --- Data Handling Functions ---

def create_multi_horizon_sequences(data, target_data, seq_length, day_horizon, week_horizon):
    """ Creates input sequences (X) and corresponding multi-horizon target sequences. """
    X, y_day, y_week = [], [], []
    max_horizon = max(day_horizon, week_horizon)
    # Ensure enough data for at least one sequence + longest horizon
    if len(data) < seq_length + max_horizon:
        st.error(f"Insufficient data length ({len(data)}) after processing to create sequences. "
                 f"Need at least {seq_length + max_horizon} rows.")
        return np.array([]), np.array([]), np.array([])

    # Iterate through the data to create sequences
    num_sequences = len(data) - seq_length - max_horizon + 1
    if num_sequences <= 0:
        st.error(f"Cannot create sequences with current data length ({len(data)}), sequence length ({seq_length}), "
                 f"and max horizon ({max_horizon}).")
        return np.array([]), np.array([]), np.array([])

    for i in range(num_sequences):
        X.append(data[i : i + seq_length])
        y_day.append(target_data[i + seq_length : i + seq_length + day_horizon])
        y_week.append(target_data[i + seq_length : i + seq_length + week_horizon])

    return np.array(X), np.array(y_day), np.array(y_week)

# --- Data Preprocessing (MODIFIED: Returns last timestamp) ---
def load_and_prepare_data(file_path, required_columns, target_columns):
    """
    Loads data, gets last timestamp, validates/resamples, preprocesses, normalizes.
    Returns: normalized input data, normalized target data, scalers, last_known_timestamp.
    """
    last_known_timestamp = None # Initialize
    try:
        if hasattr(file_path, 'getvalue'):
            # Reset buffer if it's an UploadedFile object that might have been read
            try:
                file_path.seek(0)
            except: pass # Ignore if seek is not available or fails
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(str(file_path))
        # st.info(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns.") # Less verbose
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        return None, None, None, None

    # --- 1. Process Timestamp and Find Last Valid One ---
    if 'Timestamp' not in df.columns:
        st.error("Validation Failed: Missing required 'Timestamp' column.")
        return None, None, None, None
    try:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        original_rows = len(df)
        df.dropna(subset=['Timestamp'], inplace=True)
        rows_dropped = original_rows - len(df)
        if rows_dropped > 0:
            st.warning(f"Removed {rows_dropped} rows with invalid Timestamp formats.")
        if df.empty:
            st.error("No valid Timestamps found after conversion.")
            return None, None, None, None

        # *** GET LAST TIMESTAMP ***
        last_known_timestamp = df['Timestamp'].max()
        st.info(f"Last timestamp found in data: {last_known_timestamp}")

        df.set_index('Timestamp', inplace=True, drop=False) # Keep Timestamp column if needed
        df.sort_index(inplace=True)
    except Exception as e:
        st.error(f"Could not process 'Timestamp' column: {e}")
        return None, None, None, None # Return None for timestamp as well

    # --- 2. Frequency Validation & Resampling per Day ---
    processed_days_data = []
    unique_days = df.index.normalize().unique()
    # st.info(f"Data spans {len(unique_days)} days. Validating frequency (1440 points/day) ...") # Less verbose
    validation_or_resampling_failed = False
    # Use st.expander for potentially long validation output, keep it collapsed by default
    with st.expander("Frequency Validation Details", expanded=False):
        with st.spinner("Validating data frequency per day ..."):
            for day_date in unique_days:
                day_data = df[df.index.date == day_date.date()]
                day_point_count = day_data.shape[0]
                if day_point_count == 0: continue
                if day_point_count < 1440:
                    st.error(f"Validation Failed: Day {day_date.date()} has {day_point_count} points (requires 1440).")
                    validation_or_resampling_failed = True; break
                elif day_point_count > 1440:
                    st.write(f"Resampling day {day_date.date()} ({day_point_count} points)...")
                    try:
                        numeric_cols = day_data.select_dtypes(include=np.number).columns
                        day_resampled = day_data[numeric_cols].resample('T').mean()
                        day_resampled = day_resampled.ffill().bfill()
                        if len(day_resampled) != 1440:
                            st.error(f"Resampling Error: Day {day_date.date()} resulted in {len(day_resampled)} points.")
                            validation_or_resampling_failed = True; break
                        processed_days_data.append(day_resampled)
                    except Exception as resample_e:
                        st.error(f"Resampling Error for day {day_date.date()}: {resample_e}")
                        validation_or_resampling_failed = True; break
                else:
                    # st.write(f"Day {day_date.date()} has exactly 1440 points (OK).") # Less verbose
                    processed_days_data.append(day_data) # Append original day_data if count is correct
        if validation_or_resampling_failed:
            st.error("Frequency validation or resampling failed. Cannot proceed.")
            return None, None, None, last_known_timestamp
    # else: st.success("Frequency checks passed.") # Less verbose

    if not processed_days_data:
        st.error("No valid data days found after processing.")
        return None, None, None, last_known_timestamp
    df_processed = pd.concat(processed_days_data)
    df_processed.sort_index(inplace=True)
    # st.success(f"Frequency processing complete. Final dataset: {df_processed.shape[0]} rows.") # Less verbose

    # --- 3. Derive 'Day' column (0-6) AFTER resampling ---
    # Use index which is the Timestamp after resampling
    df_processed['Day'] = df_processed.index.dayofweek

    # --- 4. Check Required Columns exist AFTER potential resampling ---
    # Ensure P1-P3, Q1-Q3, Day are present
    final_check_cols = required_columns[:] # Copy list
    missing_cols = [col for col in final_check_cols if col not in df_processed.columns]
    if missing_cols:
        # Check if original df had them before resampling potentially dropped non-numeric cols
        orig_missing = [col for col in final_check_cols if col not in df.columns and col != 'Day']
        if orig_missing:
            st.error(f"Input Error: Original CSV missing required numeric columns: {', '.join(orig_missing)}")
        else:
            st.error(f"Processing Error: Missing required columns after resampling/processing: {', '.join(missing_cols)}. Ensure they were numeric.")
        return None, None, None, last_known_timestamp

    # --- 5. Select Final Columns & Final NaN Check ---
    # Select only the columns needed for the model input features
    df_final_input = df_processed[final_check_cols].copy()
    if df_final_input.isnull().values.any():
        st.warning("NaNs detected late in processing. Filling with ffill then zero.")
        df_final_input = df_final_input.ffill().fillna(0) # Fill remaining NaNs
        if df_final_input.isnull().values.any():
            st.error("Data still contains NaNs after attempting to fill. Cannot proceed.")
            return None, None, None, last_known_timestamp

    # --- 6. Normalization ---
    try:
        # Calculate scalers based on the final input data
        # Only include numeric columns for min/max calculation
        numeric_cols_for_scaling = df_final_input.select_dtypes(include=np.number).columns
        data_min = df_final_input[numeric_cols_for_scaling].min()
        data_max = df_final_input[numeric_cols_for_scaling].max()
        data_range = (data_max - data_min).replace(0, 1) # Avoid division by zero

        # Apply normalization only to numeric columns found
        df_normalized = df_final_input.copy() # Start with a copy
        for col in data_range.index: # Iterate through columns with valid range
            df_normalized[col] = (df_final_input[col] - data_min[col]) / data_range[col]


        # Prepare outputs
        # Scalers should include all required input columns if they were numeric
        scalers = {col: {'min': df_final_input[col].min(), 'max': df_final_input[col].max()}
                   for col in final_check_cols if col in df_final_input.select_dtypes(include=np.number).columns}


        # Ensure df_normalized has all required columns before creating numpy array
        if not all(col in df_normalized.columns for col in required_columns):
            missing_norm_cols = [col for col in required_columns if col not in df_normalized.columns]
            st.error(f"Normalization Error: Missing columns after normalization: {', '.join(missing_norm_cols)}")
            return None, None, None, last_known_timestamp

        # Convert to numpy array using the final required column order
        data_normalized_all = df_normalized[required_columns].values.astype(np.float32)

        # Ensure target columns exist in the normalized df before selecting
        if not all(col in df_normalized.columns for col in target_columns):
            st.error(f"Target columns ({', '.join(target_columns)}) not found after normalization.")
            return None, None, None, last_known_timestamp
        data_normalized_target = df_normalized[target_columns].values.astype(np.float32)

        # st.success("Data loading, validation, processing, and normalization complete.") # Less verbose
        # *** MODIFIED RETURN ***
        return data_normalized_all, data_normalized_target, scalers, last_known_timestamp
    except Exception as norm_e:
        st.error(f"Error during data normalization: {norm_e}")
        return None, None, None, last_known_timestamp
#----------------------------------------------------------------------------------------------------------------------------------------------


class MultiTargetDataset(Dataset):
    """ Custom PyTorch Dataset for multi-horizon forecasting targets. """
    def __init__(self, X, y_day, y_week):
        self.X = X
        self.y_day = y_day
        self.y_week = y_week
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        # Return data as tensors for DataLoader
        return torch.tensor(self.X[idx], dtype=torch.float32), \
               torch.tensor(self.y_day[idx], dtype=torch.float32), \
               torch.tensor(self.y_week[idx], dtype=torch.float32)

# --- Forecasting Function ---
def forecast_multi_horizon(model, initial_data_sequence, device):
    """ Generates fixed 1-day and 7-day forecasts using the trained GRU model. """
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations for inference
        input_tensor = torch.tensor(initial_data_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        forecast_day_tensor, forecast_week_tensor = model(input_tensor)
        forecast_day_np = forecast_day_tensor.squeeze(0).cpu().numpy()
        forecast_week_np = forecast_week_tensor.squeeze(0).cpu().numpy()
    return forecast_day_np, forecast_week_np


# --- Streamlit UI Sections ---

def forecasting_ui():
    """ Defines the Streamlit UI for the Forecasting section. """
    st.header("Generate Forecasts")
    st.markdown("""
    **Data Requirement:** Upload a CSV file with historical data.
    - Include ‘Timestamp‘ and target columns ( ‘P1‘ - ‘P3‘, ‘Q1‘ - ‘Q3‘).
    - Data should ideally have a **consistent 1-minute frequency**. The system validates/resamples to 1440 points per day.
    """)

    uploaded_forecast_file = st.file_uploader("Upload CSV with Historical Data", type=["csv"], key="forecast_uploader")
    forecast_option = st.radio("Select Forecast Horizon:", ["Next Day", "Next Week"], key="forecast_horizon_choice", horizontal=True)

    if uploaded_forecast_file:
        st.info(f"Model requires last {SEQUENCE_LENGTH} data points as input.")

        # *** MODIFIED CALL: Receive last_known_timestamp ***
        data_norm_all, _, scalers, last_known_timestamp = load_and_prepare_data(
            uploaded_forecast_file,
            REQUIRED_INPUT_COLUMNS,
            TARGET_COLUMNS
        )

        if data_norm_all is None or scalers is None:
            st.error("Failed to load/process forecast data. Check messages above.")
            return # Use return instead of st.stop() within function

        if len(data_norm_all) < SEQUENCE_LENGTH:
            st.error(f"Insufficient data after processing. Need >= {SEQUENCE_LENGTH} rows, got {len(data_norm_all)}.")
            return

        initial_sequence = data_norm_all[-SEQUENCE_LENGTH:]
        data_available = True
        # st.success("Data ready for forecasting.") # Less verbose

        # --- Check if a valid last timestamp was found ---
        if pd.isna(last_known_timestamp):
            st.warning("Could not determine last timestamp from file. Forecast time reference will default to current time.")
            reference_time = datetime.now() # Fallback reference
            last_day_numeric = reference_time.weekday()
            using_fallback_time = True
        else:
            reference_time = last_known_timestamp # Use actual last timestamp
            last_day_numeric = reference_time.weekday()
            using_fallback_time = False

    else:
        st.warning("Upload CSV with historical data for forecasting.")
        data_available = False
        # return # Allow UI elements below button to render

    # Proceed only if data is available
    if data_available:
        if forecast_option == "Next Day":
            horizon_steps = DAY_HORIZON
            horizon_label = "Next Day"
            horizon_file_part = "next_24_hours"
        else: # Next Week
            horizon_steps = WEEK_HORIZON
            horizon_label = "Next Week"
            horizon_file_part = "next_7_days"

        if st.button(f"Generate Forecast for {horizon_label}", key="generate_forecast_button"):
            with st.spinner(f"Generating {horizon_label} forecast ..."):
                forecast_day_norm, forecast_week_norm = forecast_multi_horizon(
                    model,
                    initial_sequence,
                    device
                )
                if forecast_option == "Next Day":
                    selected_forecast_norm = forecast_day_norm
                else:
                    selected_forecast_norm = forecast_week_norm

            st.success(f"Forecast for {horizon_label} generated!")

            # --- Denormalization ---
            try:
                forecast_df_denormalized = pd.DataFrame(
                    selected_forecast_norm, columns=TARGET_COLUMNS)
                for col in TARGET_COLUMNS:
                    if col in scalers:
                        min_val = scalers[col]['min']
                        max_val = scalers[col]['max']
                        range_val = max_val - min_val
                        if range_val != 0:
                            forecast_df_denormalized[col] = \
                                forecast_df_denormalized[col] * range_val + min_val
                        else:
                            forecast_df_denormalized[col] = min_val
                    else: st.warning(f"Scaler info missing for '{col}'.")
                for col in TARGET_COLUMNS:
                    forecast_df_denormalized[col] = forecast_df_denormalized[col].round(4)
            except Exception as denorm_e:
                st.error(f"Error during forecast denormalization: {denorm_e}")
                return # Stop processing this forecast if denorm fails

            # *** MODIFIED: Day Labels & Timestamps BASED ON LAST DATA POINT (or fallback) ***
            days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            # 'reference_time' and 'last_day_numeric' are set based on availability of last_known_timestamp

            forecast_title = ""
            xlabel_plot = f"Time Steps (from start of forecast)" # Default x-axis label
            plot_start_time = None

            try:
                if forecast_option == "Next Day":
                    # Start plot / timestamps at 00:00 of the day AFTER the reference time's day
                    next_day_date = reference_time.date() + timedelta(days=1)
                    plot_start_time = datetime.combine(next_day_date, time(0, 0)) # Start at midnight
                    next_day_index = (last_day_numeric + 1) % 7
                    forecast_day_label = days_of_week[next_day_index]
                    ref_time_str = reference_time.strftime('%Y-%m-%d %H:%M')
                    forecast_title = f"24h Forecast (For {forecast_day_label}, starting {plot_start_time.strftime('%Y-%m-%d %H:%M')})"
                    if using_fallback_time: forecast_title += f" - Ref: Current Time {ref_time_str}"
                    else: forecast_title += f" - Ref: Data End {ref_time_str}"

                else: # Next Week
                    # Start plot / timestamps 1 minute after the reference time
                    plot_start_time = reference_time + timedelta(minutes=1)
                    start_day_index = (last_day_numeric + 1) % 7
                    forecast_days = [days_of_week[(start_day_index + i) % 7] for i in range(7)]
                    ref_time_str = reference_time.strftime('%Y-%m-%d %H:%M')
                    forecast_title = f"7-Day Forecast ({forecast_days[0]} to {forecast_days[-1]}, starting {plot_start_time.strftime('%Y-%m-%d %H:%M')})"
                    if using_fallback_time: forecast_title += f" - Ref: Current Time {ref_time_str}"
                    else: forecast_title += f" - Ref: Data End {ref_time_str}"


                # Generate timestamp range for the forecast period with 1-minute frequency
                forecast_timestamps = pd.date_range(start=plot_start_time,
                                                    periods=horizon_steps, freq='T')

                # Add timestamp column to the DataFrame
                forecast_df_denormalized['Timestamp'] = forecast_timestamps
                cols = ['Timestamp'] + TARGET_COLUMNS
                forecast_df_denormalized = forecast_df_denormalized[cols]
                xlabel_plot = "Timestamp" # Update plot label

            except Exception as e:
                st.warning(f"Could not generate accurate timestamps/labels: {e}. Plot will use time steps.")
                # If title wasn't set due to error, set a default
                if not forecast_title: forecast_title = f"{horizon_label} Forecast"


            # --- Display Results ---
            st.subheader("Denormalized Forecast Sample")
            st.dataframe(forecast_df_denormalized.head())

            # --- Plotting ---
            st.subheader("Forecast Visualization (Denormalized)")
            try:
                fig, ax = plt.subplots(figsize=(14, 7))
                x_axis = forecast_df_denormalized['Timestamp'] if 'Timestamp' in forecast_df_denormalized else forecast_df_denormalized.index
                for col in TARGET_COLUMNS:
                    if col in forecast_df_denormalized:
                        ax.plot(x_axis, forecast_df_denormalized[col], label=col)
                ax.legend()
                ax.set_xlabel(xlabel_plot)
                ax.set_ylabel("Predicted Values")
                ax.set_title(forecast_title) # Use updated title
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as plot_e:
                st.error(f"Error generating forecast plot: {plot_e}")


            # --- Download Button ---
            ref_time_str_file = reference_time.strftime('%Y%m%d_%H%M')
            csv_filename = f"forecast_{horizon_file_part}_ref_{ref_time_str_file}.csv"
            try:
                csv_data = forecast_df_denormalized.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"Download Forecast ({horizon_label})",
                    data=csv_data,
                    file_name=csv_filename,
                    mime="text/csv",
                    key="download_forecast_button"
                )
            except Exception as e:
                st.error(f"Failed to prepare download link: {e}")


def training_ui():
    """ Defines the Streamlit UI for the Model Training section. """
    st.header("Train Multi-Horizon GRU Model")
    st.markdown("""
    **Data Requirement:** Upload a CSV file with historical data.
    - Include ‘Timestamp‘ and target columns ( ‘P1‘ - ‘P3‘, ‘Q1‘ - ‘Q3‘).
    - Data must have a **consistent 1-minute frequency (1440 points per day)** after validation/resampling.
    """)

    uploaded_file = st.file_uploader("Upload CSV for Model Training", type=["csv"], key="train_uploader")

    with st.expander("Set Training Parameters"):
        num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=10, step=1, key="train_epochs")
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=1e-1, value=0.001, step=1e-4, format="%.4f", key="train_lr")
        batch_size = st.selectbox("Batch Size", options=[8, 16, 32, 64], index=1, help="Number of sequences per training batch.", key="train_batch_size")

    if uploaded_file:
        if st.button("Train GRU Model", key="train_model_button"):
            st.info("Loading and preparing training data ...")
            # Load and preprocess - Ignore last timestamp return value for training
            data_norm_all, data_norm_target, _, _ = load_and_prepare_data(
                uploaded_file,
                REQUIRED_INPUT_COLUMNS,
                TARGET_COLUMNS
            )

            if data_norm_all is None or data_norm_target is None:
                st.error("Failed to load/process training data.")
                return # Use return instead of st.stop()

            st.info("Creating training sequences ...")
            X_train, y_day_train, y_week_train = create_multi_horizon_sequences(
                data_norm_all,
                data_norm_target,
                SEQUENCE_LENGTH,
                DAY_HORIZON,
                WEEK_HORIZON
            )

            if X_train.shape[0] == 0: return # Error handled in function

            st.success(f"Created {X_train.shape[0]} training sequences.")

            train_dataset = MultiTargetDataset(X_train, y_day_train, y_week_train)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

            model_to_train = model # Use globally loaded model
            model_to_train.to(device)
            optimizer = optim.Adam(model_to_train.parameters(), lr=learning_rate)

            st.info(f"Starting GRU training for {num_epochs} epochs ...")
            model_to_train.train()
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            loss_history = []
            chart_placeholder = st.empty()
            training_successful = True

            try:
                for epoch in range(num_epochs):
                    epoch_loss = 0.0
                    processed_batches = 0
                    num_batches = len(train_dataloader)
                    for i, batch_data in enumerate(train_dataloader):
                        try:
                            inputs, target_day, target_week = [d.to(device) for d in batch_data]
                            out_day, out_week = model_to_train(inputs)
                            loss_day = criterion(out_day, target_day)
                            loss_week = criterion(out_week, target_week)
                            loss = loss_day + loss_week
                            if torch.isnan(loss) or torch.isinf(loss):
                                continue
                            optimizer.zero_grad(); loss.backward(); optimizer.step()
                            epoch_loss += loss.item(); processed_batches += 1
                        except Exception as batch_e:
                            st.warning(f"Batch {i+1} error: {batch_e}"); continue

                    if processed_batches > 0:
                        avg_epoch_loss = epoch_loss / processed_batches
                        loss_history.append(avg_epoch_loss)
                        progress = (epoch + 1) / num_epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_epoch_loss:.6f}")
                        loss_df = pd.DataFrame({'Epoch': range(1, epoch + 2), 'Average Loss': loss_history})
                        chart_placeholder.line_chart(loss_df.set_index('Epoch'))
                    else: st.warning(f"Epoch {epoch +1} had no processed batches.")

            except Exception as train_e:
                st.error(f"Training loop error: {train_e}"); training_successful = False

            progress_bar.empty()
            if training_successful and loss_history:
                status_text.text(f"Training finished. Final Avg Loss: {loss_history[-1]:.6f}")
                st.success("GRU Model training completed!")
                st.info(f"Saving trained model weights to {MODEL_PATH}... ")
                try:
                    model_to_train.to('cpu')
                    torch.save(model_to_train.state_dict(), MODEL_PATH)
                    st.success(f"Model weights saved successfully!")
                    model_to_train.to(device)
                    st.cache_resource.clear(); st.info("Model cache cleared.")
                except Exception as e: st.error(f"Error saving model: {e}")
            elif training_successful: status_text.text("Training loop finished, but no loss recorded.")
            else: status_text.text("Training stopped due to error.")
    else:
        st.info("Upload CSV to enable model training.")


# --- Main App Structure for this page ---
tab1, tab2 = st.tabs(["Generate Forecasts", "Train Model"])
with tab1: forecasting_ui()
with tab2: training_ui()
