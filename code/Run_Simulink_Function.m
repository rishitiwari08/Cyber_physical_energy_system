function [a] = Run_Simulink_Function(b, c)
    mdl = b;
    x = c;
    % Open and configure the Simulink model
    open_system(mdl);
    % Set the value of the Constant Block (assuming block path is 'mdl/ConstantBlockName')
    set_param([mdl, '/Constant'], 'Value', num2str(x)); % A_value is the value you want to assign
    % Configure simulation parameters
    simIn = Simulink.SimulationInput(mdl);
    simIn = setModelParameter(simIn, 'StopTime', '5');
    simIn = setModelParameter(simIn, 'ZeroCrossControl', 'disable');
    % Run the simulation
    out = sim(simIn);
    % Extract simulation results
    outputV = out.V1ind;
    timecolumnV = outputV.Time;
    voltage1_data = out.V1ind.Data;
    current1_data = out.I1ind.Data;
    active1_data = out.P1.Data;
    reactive1_data = out.Q1.Data;
    apparent1_data = out.A1.Data;
    PF1_data = out.PF1.Data;
    voltage2_data = out.V2ind.Data;
    current2_data = out.I2ind.Data;
    active2_data = out.P2.Data;
    reactive2_data = out.Q2.Data;
    apparent2_data = out.A2.Data;
    PF2_data = out.PF2.Data;
    voltage3_data = out.V3ind.Data;
    current3_data = out.I3ind.Data;
    active3_data = out.P3.Data;
    reactive3_data = out.Q3.Data;
    apparent3_data = out.A3.Data;
    PF3_data = out.PF3.Data;
    % Create the output matrix
    OutMatrix = [timecolumnV voltage1_data voltage2_data voltage3_data ...
                 current1_data current2_data current3_data ...
                 active1_data active2_data active3_data ...
                 reactive1_data reactive2_data reactive3_data ...
                 apparent1_data apparent2_data apparent3_data ...
                 PF1_data PF2_data PF3_data];
    a = OutMatrix;
end