import json
import os
import tensorflow as tf
from bowman.training_run import TrainingRun

config_dir = "configs/bowman/big_experiment4/"
print("begin")
print(os.listdir(config_dir))

for experiment_config in os.listdir(config_dir):
    with open(config_dir + experiment_config, 'r') as f:
        config = json.load(f)['config']

    print("----------------------------------------")
    print("Experiment: " + experiment_config)
    print(json.dumps(config, indent=4, sort_keys=True))
    print("----------------------------------------")
    run = TrainingRun(config, False, experiment_config + ".txt")
    run.train(restore_state=False)
    tf.reset_default_graph()

#f.close()