from bowman.training_run import TrainingRun
import json


def interact_from_prompt(run, config):
    s = input("Enter a sentence or x to cancel: ")
    out = open(config["log_dir"] + "interact.txt", "w")
    while s != "x":
        run.interact([s], out, 0.1)
        s = input("Enter a sentence or x to cancel: ")
    out.close()

def interact_from_list(run, sentences, config):
    out = open(config["log_dir"] + "interact.txt", "w")
    run.interact(sentences, out, 0.3)
    out.close()

def generate_random_sentences(run, config):
    out = open(config["log_dir"] + "interact.txt", "w")
    run.generate_random_sentences(0, 1, out)
    out.close()

def get_metrics(run):
    run.test_model(32, "interact.txt")

def interpolate(run):
    run.interpolate_sentences(4, "interpolate.txt")

def gen_with_noise(run):
    run.gen_sentences_noise(4, "gen_with noise.txt", 0.003)


if __name__ == "__main__":
    with open("../configs/bowman/best_model.json", 'r') as f:
        config = json.load(f)['config']

    run = TrainingRun(config, True, "output.txt")

    get_metrics(run)#interact_from_list(run, ["i liked the ice cream lots"], config)
    #generate_random_sentences(run, config)

