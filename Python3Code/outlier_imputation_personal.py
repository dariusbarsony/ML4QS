import pandas as pd
import numpy as np
import sys
import copy
import matplotlib.pyplot as plt
from util.VisualizeDataset import VisualizeDataset

sys.path.append(".")

from Chapter3.OutlierDetection import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection

df = pd.read_csv('intermediate_datafiles/chapter2_result.csv')
print(df['acc_phone_x'])

DistOut = DistributionBasedOutlierDetection()
DistanceOut = DistanceBasedOutlierDetection()

C = [0.5,0.75,1,2,3,4]

DataViz = VisualizeDataset(__file__)
col = 'acc_phone_x'

# Chauvenet

# for c in C:
#     dataset = DistOut.chauvenet(df,col,c)
#     print(c)
#     DataViz.plot_binary_outliers(
#         dataset, col, col + '_outlier')

# Mixture

N = [1,2,3,4,5]

# Mixture

# for n in N:
#     dataset = DistOut.mixture_model(df, col,n)
#     DataViz.plot_dataset(dataset, [
#                              col, col + '_mixture'], ['exact', 'exact'], ['line', 'points'])

# Simple distance based

# d_min = [0.05,0.1,0.2]
# f_min = [0.98,0.99,0.995]
#
# for d in d_min:
#     for f in f_min:
#         try:
#             dataset = DistanceOut.simple_distance_based(
#                 df, [col], 'euclidean', d, f)
#             DataViz.plot_binary_outliers(
#                 dataset, col, 'simple_dist_outlier')
#         except MemoryError as e:
#             print(
#                 'Not enough memory available for simple distance-based outlier detection...')
#             print('Skipping.')

# Local outlier factor

K = [3,5,7]

for k in K:
    
