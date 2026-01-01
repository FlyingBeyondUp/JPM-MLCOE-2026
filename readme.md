# Dependencies
tensorflow                 2.20.0 
tensorflow-probability     0.25.0 
numpy                      2.4.0rc1  
matplotlib                 3.10.7 
scipy                      1.16.3 

# Usage
## Run KF demo
1.`python linearSSM.py`
2. `python checkKF.py`

## Run EKF and UKF demo
1.`python nonlinearSSM_saved.py`

## Run PF demo'
`python particle_filter.py`

## Run Particle Flow demo
`python particle_flow_filters_saved.py`

## Compare different filters
`python compare_flow_filters.py`

# Notes
- The _saved suffix indicates the old version that can reproduce figures in the report.
- Further study is needed for the Invertible Particle Flow method as it shows unstable performance in some runs.
