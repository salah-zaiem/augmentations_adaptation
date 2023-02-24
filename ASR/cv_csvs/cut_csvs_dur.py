import os
import pandas as pd
import sys
import numpy as np 

dirin = sys.argv[1]
duration = [600,1800,3600,7200]

train_dir  = os.path.join(dirin, "train.csv")
df = pd.read_csv(train_dir)
print(df.head())
durations = list(df["duration"])
indexes = []
print(np.cumsum(durations))
print(f"total duration : {np.sum(durations)}")
i=0
for ind, val in enumerate(np.cumsum(durations)) : 
    if i< len(duration):
        if val > duration[i] : 
            indexes.append(ind)
            i+=1
print(len(durations))
#print(indexes)
fileids = list(df["ID"])
for i in range(len(indexes)):
    associated_files = (fileids[0:indexes[i]])
    df_here = df[(df["ID"]).isin(associated_files)]
    dur = duration[i]
    #df_here.to_csv(os.path.join(dirin, f"train_{dur}.csv"))

