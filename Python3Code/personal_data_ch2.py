##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 2                                               #
#                                                            #
##############################################################

# Import the relevant classes.
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys


# Chapter 2: Initial exploration of the dataset.

DATASET_PATH = Path('./datasets/final_assignment/')
RESULT_PATH = Path('./results/')
RESULT_FNAME = 'ch2_result.csv'

# Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
# instance per minute, and a fine-grained one with four instances per second.
GRANULARITIES = [250]

# We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

print('Please wait, this will take a while to run!')

datasets = []

for milliseconds_per_instance in GRANULARITIES:
    print(f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

    # Create an initial dataset object with the base directory for our data and a granularity
    dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

    # Add the selected measurements to it.

    # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
    # and aggregate the values per timestep by averaging the values
    # dataset.create_label_file('carlitos.csv', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'classe')

    # dataset.add_numerical_dataset('carlitos.csv', 'timestamps', ["num_window","roll_belt","pitch_belt","yaw_belt","total_accel_belt","max_roll_belt","max_picth_belt","min_roll_belt","min_pitch_belt","amplitude_roll_belt","amplitude_pitch_belt","var_total_accel_belt","avg_roll_belt","stddev_roll_belt","var_roll_belt","avg_pitch_belt","stddev_pitch_belt","var_pitch_belt","avg_yaw_belt","stddev_yaw_belt","var_yaw_belt","gyros_belt_x","gyros_belt_y","gyros_belt_z","accel_belt_x","accel_belt_y","accel_belt_z","magnet_belt_x","magnet_belt_y","magnet_belt_z","roll_arm","pitch_arm","yaw_arm","total_accel_arm","var_accel_arm","avg_roll_arm","stddev_roll_arm","var_roll_arm","avg_pitch_arm","stddev_pitch_arm","var_pitch_arm","avg_yaw_arm","stddev_yaw_arm","var_yaw_arm","gyros_arm_x","gyros_arm_y","gyros_arm_z","accel_arm_x","accel_arm_y","accel_arm_z","magnet_arm_x","magnet_arm_y","magnet_arm_z","max_roll_arm","max_picth_arm","max_yaw_arm","min_roll_arm","min_pitch_arm","min_yaw_arm","amplitude_roll_arm","amplitude_pitch_arm","amplitude_yaw_arm","roll_dumbbell","pitch_dumbbell","yaw_dumbbell","max_roll_dumbbell","max_picth_dumbbell","min_roll_dumbbell","min_pitch_dumbbell","amplitude_roll_dumbbell","amplitude_pitch_dumbbell","total_accel_dumbbell","var_accel_dumbbell","avg_roll_dumbbell","stddev_roll_dumbbell","var_roll_dumbbell","avg_pitch_dumbbell","stddev_pitch_dumbbell","var_pitch_dumbbell","avg_yaw_dumbbell","stddev_yaw_dumbbell","var_yaw_dumbbell","gyros_dumbbell_x","gyros_dumbbell_y","gyros_dumbbell_z","accel_dumbbell_x","accel_dumbbell_y","accel_dumbbell_z","magnet_dumbbell_x","magnet_dumbbell_y","magnet_dumbbell_z","roll_forearm","pitch_forearm","yaw_forearm","max_roll_forearm","max_picth_forearm","min_roll_forearm","min_pitch_forearm","amplitude_roll_forearm","amplitude_pitch_forearm","total_accel_forearm","var_accel_forearm","avg_roll_forearm","stddev_roll_forearm","var_roll_forearm","avg_pitch_forearm","stddev_pitch_forearm","var_pitch_forearm","avg_yaw_forearm","stddev_yaw_forearm","var_yaw_forearm","gyros_forearm_x","gyros_forearm_y","gyros_forearm_z","accel_forearm_x","accel_forearm_y","accel_forearm_z","magnet_forearm_x","magnet_forearm_y","magnet_forearm_z"], 'avg', '')
    dataset.add_numerical_dataset('carlitos.csv', 'timestamps', ['yaw_dumbbell', 'pitch_dumbbell', 'roll_dumbbell', 'magnet_forearm_z', 'magnet_arm_z', 'magnet_arm_y', 'magnet_arm_x', 'accel_arm_z', 'accel_arm_y', 'accel_arm_x', 'gyros_arm_z', 'gyros_arm_y', 'gyros_arm_x', 'total_accel_arm', 'yaw_arm', 'pitch_arm', 'roll_arm', 'total_accel_dumbbell', 'gyros_dumbbell_x', 'gyros_dumbbell_y', 'gyros_dumbbell_z', 'magnet_forearm_y', 'magnet_forearm_x', 'accel_forearm_z', 'accel_forearm_y', 'accel_forearm_x', 'gyros_forearm_z', 'gyros_forearm_y', 'gyros_forearm_x', 'magnet_belt_z', 'total_accel_forearm', 'pitch_forearm', 'roll_forearm', 'magnet_dumbbell_z', 'magnet_dumbbell_y', 'magnet_dumbbell_x', 'accel_dumbbell_z', 'accel_dumbbell_y', 'accel_dumbbell_x', 'yaw_forearm', 'magnet_belt_y', 'accel_belt_z', 'total_accel_belt', 'magnet_belt_x', 'pitch_belt', 'num_window', 'roll_belt', 'gyros_belt_x', 'gyros_belt_y', 'gyros_belt_z', 'accel_belt_x', 'accel_belt_y', 'yaw_belt'], 'avg', '')    # We add the labels provided by the users. These are categorical events that might overlap. We add them
    # as binary attributes (i.e. add a one to the attribute representing the specific value for the label if it
    # occurs within an interval).
    # dataset.create_label_file('carlitos.csv', "raw_timestamp_part_1", "raw_timestamp_part_2", 'classe')
    dataset.add_event_dataset('labels.csv', 'label_start', 'label_end', 'label', aggregation='binary')

    # Get the resulting pandas data table
    dataset = dataset.data_table

    # Plot the data
    DataViz = VisualizeDataset(__file__)

    dataset = dataset.astype(float)

    # Boxplot
    DataViz.plot_dataset_boxplot(dataset, ["num_window","roll_belt","pitch_forearm"])

    # # Plot all data
    DataViz.plot_dataset(dataset, ['num_window', "roll_belt", "pitch_forearm", 'label'],
                                  ['like','like', 'like', 'like'],
                                  ['line', 'line', 'line', 'points'])

    # And print a summary of the dataset.
    util.print_statistics(dataset)
    datasets.append(copy.deepcopy(dataset))

    # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
    # dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')

# Make a table like the one shown in the book, comparing the two datasets produced.

# util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

# Finally, store the last dataset we generated (250 ms).
dataset.to_csv(RESULT_PATH / RESULT_FNAME)

# Lastly, print a statement to know the code went through
print('The code has run through successfully!')
