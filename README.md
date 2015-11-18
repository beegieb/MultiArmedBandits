# MultiArmedBandits
Collection of Multi Armed Bandit algorithms and a Simulator 

Required packages:
```
brewer2mpl==1.4.1
funcsigs==0.4
ggplot==0.6.8
matplotlib==1.4.3
mock==1.3.0
nose==1.3.7
numpy==1.9.3
pandas==0.16.2
patsy==0.4.0
pbr==1.8.0
pyparsing==2.0.3
python-dateutil==2.4.2
pytz==2015.6
scipy==0.16.0
six==1.9.0
statsmodels==0.6.1
wheel==0.24.0
```

Running a simulation:
```
import numpy as np

from simulation import BernoulliArm, BanditSimulation
from algorithms import RandomBetaBandit

# Simulate 30 bernoulli arms
arms = [BernoulliArm(p=p) for p in np.linspace(0.1, 0.9, 30)]

# Initialize the bandit algorithm, we'll use the RandomBetaBandit (also known as a Baysean Bandit)
bandit = RandomBetaBandit(n_arms=30)

# Initialize the simulator
sim = BanditSimulation(arms=arms,
                       n_rounds=5000,  # the number of rounds we'll run per simulation
                       n_sim=100,      # the number of simulations we'll run
                       delay=0,        # how much we delay feedback when updating the bandit
                       verbose=False,  # should we print runtime results?
                       outfile=None)   # an optional file to save the simulation results 
                       
# Run the simulation                       
sim.simulate(bandit)

# Summarize the results of the simulation
sim.print_summary()

# plot results (requires the ggplot package)
# what='all' will plot accuracy plot and cumulative payouts
# include_ci=True includes plots of confidence intervals 
plt = sim.plot(what='all', include_ci=True) 

from ggplot import ggsave
ggsave(plt, 'results.png')

# save simulation results to CSV for later analysis
sim.save_results('results.csv')
```