import numpy as np
import pandas as pd
import os
import ntpath

rid_targets = {}
def find_rid_targets(dirname):

    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                find_rid_targets(full_filename)
            else:

                filename = ntpath.basename(full_filename)
                rId = os.path.splitext(filename)[0]
                ext = os.path.splitext(filename)[-1]

                if ext != '.csv':
                    continue

                files = rid_targets.get(rId)
                if files:
                    files.append(full_filename)
                else:
                    rid_targets[rId] = [full_filename]

    except PermissionError:
        pass

find_rid_targets("../data/")

timeseries = []
i = 0
for n in rid_targets:
    j = 0
    for p in rid_targets[n]:
        data = pd.read_csv(p, index_col=0, parse_dates=True)
        data_kw = data.resample('1H').sum()

        if j == 0:
            timeseries.append(np.trim_zeros(data_kw.iloc[:, 0], trim='f'))
        else:
            timeseries[i] = timeseries[i].append(np.trim_zeros(data_kw.iloc[:, 0], trim='f'))
        j += 1
    i += 1


print(timeseries)
print(timeseries[0])
