import pandas as pd

data = pd.read_csv("../data/R000000002A6_CYCLE_ACTIVE_ENERGY.csv", index_col=0, parse_dates=True)
data_kw = data.resample('2H').sum()

print(data_kw)