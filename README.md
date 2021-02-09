# mope
martingale off-policy evaluation

# conda installation
conda env create -f environment.yml

# outline
- `opebet.py` contains the code for MOPE (`wealth_lb_2d`) and its 
ablations:
  - `wealth_lb_1d`: scalar betting
  - `wealth_2d`: exact wealth maximization
  - `wealth_lb_2d_individual_qps`: individual bets per value on a grid
- `opebetrp.py` contains code for reward predictors and gated deployment
  - `wealth_lb_rp` subtracts the reward predictor control variate from w*r  
  - `wealth_lb_rp_double_hedge` the double hedging strategy
  - `wealth_lb_gd` confidence sequence for gated deployment
