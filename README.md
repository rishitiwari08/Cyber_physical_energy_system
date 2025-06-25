#  Cyber-Physical Energy Management System (CPEMS)

This repository contains a cyber-physical simulation and forecasting platform for **industrial energy consumption**, integrating **MATLAB/Simulink**, **Python**, **InfluxDB**, and a **Streamlit-based HMI**. The system simulates industrial load profiles and uses a **GRU neural network** for intelligent load forecasting.

---

##  Authors  
- Jigisha Sanjay Ahirrao  
- Kumpal Ganeshbhai Khokhariya  
- Milankumar Hareshbhai Mavani  
- **Rishikesh Ravikiran Tiwari**  
- Niels Schwung  

---

##  Project Overview  

The CPEMS system provides:

-  **Simulink Simulation**: Industrial loads like motors, furnaces, resistive loads with harmonic distortion  
-  **Load Forecasting**: GRU neural network to forecast future active/reactive power  
-  **Time-Series Storage**: Uses InfluxDB Cloud  
-  **User Interface**: Streamlit dashboard for control and visualization  

---

##  Tech Stack  

| Component        | Technology                     |
|------------------|--------------------------------|
| Simulation       | MATLAB + Simulink              |
| Backend Logic    | Python, NumPy, Pandas           |
| Forecasting      | PyTorch (GRU), Streamlit       |
| Data Storage     | InfluxDB Cloud                 |
| Visualization    | Streamlit, Matplotlib          |

---

##  System Architecture  

```xml
graph TD
    A[User] --> B(Streamlit HMI);
    B --> C(MATLAB Engine - Python);
    C --> D(Run_Simulink_Function.m);
    D --> E(Simulink Model);
    E --> D;
    D --> F(Python Data Processing);
    F --> G[InfluxDB];
    G --> B;
    B --> H(GRU Forecasting - Python);
    H --> B;
```

## Folder Structure

```xml

code/
├── pages/
│   ├── 1_Forecasting.py
│   └── 2_Simulation.py
├── Run_Simulink_Function.m
├── Switch.slx
├── main.py
└── multiOutputSwitch.m

report/
├── CPEMS_Case_Study_Documentation.pdf

```


## Software and Dependencies

To run the system, you will need:

- Python (version 3.9 or higher):

- `streamlit`

- `pandas`

- `numpy`

- `torch` (for GRU model)

- `matplotlib` (for plotting)

- `influxdb-client` (for InfluxDB interaction)

- `scikit-learn` (optional, but good for data preprocessing if used)

- MATLAB (with Simulink): Required for running the simulation model.

- MATLAB Engine API for Python: Enables the Python application to interact with MATLAB.

- InfluxDB Cloud (Free Tier): A cloud-hosted time-series database for data storage. You will need to configure your URL, token, and organization.



## MATLAB Engine API Installation
From your MATLAB installation directory, run:
```xml
cd "matlabroot/extern/engines/python"
python setup.py install
```
Replace `matlabroot` with your actual MATLAB installation path.

## Steps to Run the System

1. Install MATLAB and Simulink: Ensure they are properly installed on your system.

2. Install MATLAB Engine API for Python: Follow the instructions above.
     - You can test the installation by running `python -c "import matlab.engine; eng = matlab.engine.start_matlab();"`

3. Install Python Dependencies:
```xml
pip install streamlit pandas numpy torch matplotlib influxdb-client scikit-learn
```

4. Configure InfluxDB: Update the InfluxDB URL, token (for both simulation write and visualization query), and organization parameters in `pages/2_Simulation.py` and `pages/1_Forecasting.py`.

5. Configure MATLAB Script Path: In `pages/2_Simulation.py`, ensure `DEFAULT_MATLAB_SCRIPT_PATH` points to the correct directory containing `Run_Simulink_Function.m` and `Switch.slx`.

6. Launch the Streamlit Application:
```xml
streamlit run main.py
```
This will open the HMI in your web browser, allowing you to access the Load Forecasting and Industrial Profile (Simulation) modules.

## Key Features & Implementations

- Simulink Model (`Switch.slx`):

  - Features a 3-phase controlled AC power source with custom harmonic injection (1st, 3rd, 5th, 7th harmonics).

  - Includes various industrial loads: induction motor, induction furnace, general resistive load, and pump.

  - Uses a `multiOutputSwitch` MATLAB function block to dynamically select load combinations based on user input from the HMI.

  - Incorporates power measurement blocks to calculate voltage, current, active power, reactive power, apparent power, and power factor for each phase.

  - Configured to export data to MATLAB workspace ( `To Workspace` block) for Python integration.

  - Time compression model: 24 seconds of simulation time represent 24 real-world hours, with loads scheduled according to industrial demand curves using step functions.

- MATLAB Functions:

  - `Run_Simulink_Function.m`: Called by Python to open and run the Simulink model, set the load condition, and extract simulation results into a matrix for Python.

  - `multiOutputSwitch.m`: A helper function called within Simulink to control the activation of different loads based on a numeric input condition.

- Python Logic:

  - MATLAB Engine API: Seamlessly integrates Python with MATLAB for simulation execution and data transfer.

  - Data Preprocessing: Handles timestamping, validation of data frequency (resampling to 1-minute intervals if needed), feature engineering (selecting `P1`-`P3`, `Q1`-`Q3`, `Day`), and min-max normalization.

  - InfluxDB Integration: Uses `influxdb-client` to write processed simulation data to InfluxDB and query it for visualization.

  - GRU Model (`1_Forecasting.py`):

    - A PyTorch-based Gated Recurrent Unit (GRU) neural network for multi-horizon load forecasting (next day/next week).

    - Trained using  `SmoothL1Loss` and `Adam` optimizer.

    - Features include `SEQUENCE_LENGTH` (1440 minutes/1 day) for input, `DAY_HORIZON` (1440) and `WEEK_HORIZON` (10080) for prediction.

    - Handles data loading, resampling, and normalization for both training and inference.

  - Streamlit HMI (`main.py`, `1_Forecasting.py`, `2_Simulation.py`):

    - Provides a user-friendly web interface.

    - Allows selection of machine types for simulation, execution of simulations, and real-time visualization of results.

    - Enables uploading historical data, training the GRU model, generating forecasts, and downloading forecast results.

    - Optimized for responsive layout using Streamlit's columns and markdown with HTML/CSS.

## Future Work
- Integration of Real-World Data: Extend the system to incorporate actual measurement data from industrial devices (e.g., Siemens Sentron, Beckhoff TwinCAT).

- Advanced Load Modeling: Include non-linear load effects in the Simulink model and add metrics like Total Harmonic Distortion (THD) to data analysis.

- Additional Machine Types: Expand the Simulink model to include more diverse machine types and detailed individual machine monitoring.

- Optimized Training Environment: Implement cloud-based or remote server training for the GRU model to handle larger datasets and more intensive training epochs.

- Enhanced Use Cases: Develop features for load optimization, integration of renewable energy generation (e.g., PV, battery storage), and capacity analysis for system expansion.

- Streamlined Data Flow: Integrate the forecasting module to directly read simulation data from InfluxDB for model training, eliminating the need for CSV file transfers.

## License
This project is licensed under the MIT License. See the LICENSE file for details.


