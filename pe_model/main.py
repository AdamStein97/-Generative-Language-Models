from pe_model.training_run import TrainingRun
import json


def train(run):
    print("start")
    run.train(restore_state=False)
    print("finished")

def generate(run, sentences):
    run.interact(sentences, "../data/neural_editor_data/models/trained_model.ckpt-99")

with open("../configs/PE/basic_config.json", 'r') as f:
    config = json.load(f)['config']

run = TrainingRun(config, False, "pe_fix.txt")
train(run)
