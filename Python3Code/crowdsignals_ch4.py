##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

import sys
import copy
import pandas as pd
import time
from pathlib import Path
import argparse

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

DATA_PATH = Path('./FinalAssignment/')
DATASET_FNAME = 'ch3_result.csv'
RESULT_FNAME = 'ch4_result.csv'

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    print_flags()
    
    start_time = time.time()
    try:
        dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
    except IOError as e:
        print('File not found, try to run previous crowdsignals scripts first!')
        raise e

    # Let us create our visualization class again.
    DataViz = VisualizeDataset(__file__)

    # Compute the number of milliseconds covered by an instance based on the first two rows
    milliseconds_per_instance = (dataset.index[1] - dataset.index[0]).microseconds/1000

    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    if FLAGS.mode == 'aggregation':
        # Chapter 4: Identifying aggregate attributes.

        # Set the window sizes to the number of instances representing 5 seconds, 30 seconds and 5 minutes
        window_sizes = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(5*60000)/milliseconds_per_instance)]

         #please look in Chapter4 TemporalAbstraction.py to look for more aggregation methods or make your own.     
        
        for ws in window_sizes:
            dataset = NumAbs.abstract_numerical(dataset, ['acc_phone_x'], ws, 'mean')
            dataset = NumAbs.abstract_numerical(dataset, ['acc_phone_x'], ws, 'std')

        DataViz.plot_dataset(dataset, ['acc_phone_x', 'acc_phone_x_temp_mean', 'acc_phone_x_temp_std', 'label'], ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))
  
    if FLAGS.mode == 'frequency':
        # Now we move to the frequency domain, with the same window size.
       
        fs = float(1000)/milliseconds_per_instance
        ws = int(float(10000)/milliseconds_per_instance)

        dataset = FreqAbs.abstract_frequency(dataset, ['pitch_forearm'], ws, fs)
        
        # Spectral analysis.
        DataViz.plot_dataset(dataset, ['pitch_forearm', 'label'], ['like', 'like'], ['line','points'])
        print("--- %s seconds ---" % (time.time() - start_time))
  
    if FLAGS.mode == 'final':
        
        ws = int(float(0.5*60000)/milliseconds_per_instance)
        fs = float(1000)/milliseconds_per_instance

        selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]

        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'mean')
        dataset = NumAbs.abstract_numerical(dataset, selected_predictor_cols, ws, 'std')

        # TODO: Add your own aggregation methods here
        DataViz.plot_dataset(dataset, ['pca_1', 'label'], ['exact', 'like'], ['line', 'points'])
     
        CatAbs = CategoricalAbstraction()
        
        dataset = CatAbs.abstract_categorical(dataset, ['label'], ['like'], 0.03, int(float(5*60000)/milliseconds_per_instance), 2)

        periodic_predictor_cols = ['num_window', 'roll_belt', 'pitch_forearm', 'yaw_belt', 'magnet_dumbbell_z',
             'pitch_belt', 'magnet_dumbbell_z', 'roll_forearm', 'accel_dumbbell_y', 'roll_dumbbell', 'accel_dumbbell_y']

        dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), periodic_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)

        # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

        # The percentage of overlap we allow
        window_overlap = 0.9
        skip_points = int((1-window_overlap) * ws)
        dataset = dataset.iloc[::skip_points,:]

        dataset.to_csv(DATA_PATH / RESULT_FNAME)

        DataViz.plot_dataset(dataset, ['num_window', "roll_belt", "pitch_forearm", 'pca_1', 'label'], ['like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'line', 'points'])
        print("--- %s seconds ---" % (time.time() - start_time))
  
if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help= "Select what version to run: final, aggregation or freq \
                        'aggregation' studies the effect of several aggeregation methods \
                        'frequency' applies a Fast Fourier transformation to a single variable \
                        'final' is used for the next chapter ", choices=['aggregation', 'frequency', 'final']) 

    

    FLAGS, unparsed = parser.parse_known_args()
    
    main()