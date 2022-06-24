import pandas as pd
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt

sys.path.append(".")

from Chapter3.ImputationMissingValues import ImputationMissingValues

df = pd.read_csv('intermediate_datafiles/chapter3_result_outliers.csv')

percent_missing = df['hr_watch_rate'].isnull().sum() * 100 / len(df)
print(percent_missing)


MisVal = ImputationMissingValues()
imputed_df = MisVal.impute_interpolate(copy.deepcopy(df),'hr_watch_rate')


fig=plt.figure()
ax1 = plt.subplot(2, 1, 1) # row 1, col 2 index 1
ax1.scatter(df['hr_watch_rate'], df.index)
# ax1.title("Original dataset")


ax2 = plt.subplot(2, 1, 2) # index 2
ax2.scatter(imputed_df['hr_watch_rate'], df.index)
# ax2.title("Imputed dataset")

ax1.get_shared_x_axes().join(ax1, ax2)
ax1.set_xticklabels([])
# ax2.autoscale() ## call autoscale if needed

plt.show()
