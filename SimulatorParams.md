## Simulation Parameters

Below are the parameters used in the simulation and their allowed values:

### Parameters

#### Learning and Choice Model of Humans
- **Model_Name**: 
  - Allowed values: `"Gumbel"`, `"GumbelEps"`, `"Logit"`, `"EpsGreedy"`

#### Fleet Optimization Mode
- **FleetMode**:
  - Allowed values: `"1"`, `"0"`, `"0"`, `"1+0"` (In this version, the only allowed value)

#### Human Learning Rate
- **Alpha_Zero**:
  - Range: `0.0 - 1.0`

#### Human Exploration Rate
- **Epsilon**:
  - Range: `0.0 - 1.0`

#### Dispersion Coefficient
- **Logit_Exp_Coeff (LogitParam)**:
  - Range: `0.0 - (+infinity)`

#### Initial Knowledge of Human Drivers
- **Initial_Knowledge**:
  - Allowed values: `"FreeFlow"`, `"Pessimistic"`, `"Optimistic"`

#### Initial Choice
- **Initial_Choice**:
  - Allowed values: `"Argmin"`, `"Random"`

#### Learning from Experience Only
- **Only_Experience**:
  - Allowed values: `true`, `false`

#### Fleet Size
- **Fleet_size**:
  - Range: `0.0 - 1.0` (Fraction of all vehicles which mutate into fleet)

#### Fleet Introduction Day
- **Fleet_introduction**:
  - Range: `0 - (+infinity)`

#### Fleet Optimization Onset
- **FleetOptimizationOnset**:
  - Range: `0 - (+infinity)`

#### Fleet Optimization End
- **FleetOptimizationEnd**:
  - Range: `0 - (+infinity)` (Default: last day)

#### Coefficient of the Target
- **LambdaHDV**:
  - Range: `(-infinity) - (+infinity)`
- **LambdaCAV**:
  - Range: `(-infinity) - (+infinity)`

#### CAV Target Behavior
- **CAVTarget**:
  - Allowed values: `"Selfish"`, `"Malicious"`, `"Disruptive"`, `"Social"`, `"Altruistic"`

#### Congestion Levels
- **totalVehicles**:
  - Range: `0.0 - (+infinity)` (Typically `0.25 - 2.6` times capacity)

#### Vehicle Multiplier
- **multiplier**:
  - Range: `0 - (+infinity)` (totalVehicles * multiplier prescribes the total number of vehicles)

#### Number of Simulation Days
- **dayRange**:
  - Range: `1 - (+infinity)`

#### Road Networks
- **RoadNetworks**:
  - Allowed values: `"Net2Plain"`, `"Net3twoEqual"`, `"NetBraess"`, `"NetBraess2"` (Default: one only)

#### Network Parameter Change Days
- **NetworkDayChange**:
  - Range: `1 - (+infinity)`

#### Plotting Options
- **PlotCAVs**:
  - Allowed values: `true`, `false` (Specifies whether there are CAVs to plot)
- **PlotMeanHDV**:
  - Allowed values: `true`, `false` (Specifies whether plotter should plot Mean HDV travel time)

### Obsolete Parameters
- **MovingAverageStep**: (In this version, obsolete)
