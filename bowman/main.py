from bowman.training_run import TrainingRun
import json

#run using "python -m bowman.main" from dis aux directory

with open("configs/bowman/best_model.json", 'r') as f:
    config = json.load(f)['config']

run = TrainingRun(config, True, "cauchy_tiny.txt")
print("start")
print(json.dumps(config, indent=4, sort_keys=True))
run.train(restore_state=False)
print("finished")

