from pathlib import Path

# from pysurvival.models.simulations import SimulationModel
import numpy as np
import pandas as pd

TEST_DIR = Path(__file__).parent

# Initializing the simulation model
# sim = SimulationModel( survival_distribution =  'exponential',
#                        risk_type = 'linear',
#                        censored_parameter = 30.0,
#                        alpha = 0.01,
#                        beta = 5.)

# Generating N random samples
# N = [25]

# for size in N:
#     dataset = sim.generate_data(num_samples = size, num_features=5)
#     preds = sim.predict_risk(dataset[['x_1', 'x_2', 'x_3', 'x_4', 'x_5']].to_numpy())
#     dataset.to_csv('/survival_simulation_'+str(size)+'.csv', index = False)

# pd.DataFrame(preds).to_csv(TEST_DIR+'/survival_simulation_preds_'+str(size)+'.csv', index = False)

# Preparing different data scenarios

df = pd.read_csv(TEST_DIR + "/test_data/survival_simulation_25.csv")

# sort
df.sort_values(by="time", inplace=True, ascending=True)

# make time discrete
df["time"] = df.time.round(0) + 1  # no zero time


# Scenario 0: Default

df.to_csv(TEST_DIR + "/test_data/survival_simulation_25_default.csv")


# Scenario 1: Lots of events >0.5 ratio

event = np.random.choice([0, 1], size=25, p=[0.2, 0.8])

df1 = df
df1["event"] = event

df1.to_csv(
    TEST_DIR + "/test_data/survival_simulation_25_high_event_ratio.csv", index=False
)

# Scenario 2: Low event ratio <0.05

event = np.random.choice([0, 1], size=25, p=[0.95, 0.05])

df2 = df
df2["event"] = event

df2.to_csv(
    TEST_DIR + "/test_data/survival_simulation_25_low_event_ratio.csv", index=False
)

# Scenario 3: First 5 samples no event

zeros = np.zeros(5)
ones = np.ones(20)

event = np.concatenate((zeros, ones), axis=0)

df3 = df
df3["event"] = event

df3.to_csv(
    TEST_DIR + "/test_data/survival_simulation_25_first_five_zero.csv", index=False
)

# Scenario 4: Last 5 samples no event

zeros = np.zeros(5)
ones = np.ones(20)
event = np.concatenate((ones, zeros), axis=0)

df4 = df
df4["event"] = event

df4.to_csv(
    TEST_DIR + "/test_data/survival_simulation_25_last_five_zero.csv", index=False
)

# Scenario 5: All events

event = np.ones(25)

df5 = df
df5["event"] = event

df5.to_csv(TEST_DIR + "/test_data/survival_simulation_25_all_events.csv", index=False)

# Scenario 6: No events

event = np.zeros(25)

df5 = df
df5["event"] = event

df5.to_csv(TEST_DIR + "/test_data/survival_simulation_25_no_events.csv", index=False)
