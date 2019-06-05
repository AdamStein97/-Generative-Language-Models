# Generative-Language-Models

A library using variational autoencoders to generate sentences. The repo contains two models for doing this: the Bowman model and the Prototype and Edit model.
For in-depth information about this work please read thesis.pdf. 


## Installation

```bash
pip install tensorflow
pip install tensorflow-probability
pip install nltk
```

## Data

Both models require a file containing word embeddings. This path must be specified in the config file.
Both files also require three files in tsv format: train, test and valid containing the data to train and evaluate the model.

### Bowman Model Data

Set of sentences separated by new lines.

### Prototype and Edit Model Data

A set of pairs of sentences: a source sentence and a target sentence which are separated with a tab. Each pair of sentences is separated with a new line.  

## Usage

### Training

```bash
python -m bowman.main
python -m pe_model.main
```
### Interaction

```bash
python -m bowman.interact
```

There are a number of functions in the interact file than can be used to perform different tasks with the trained model.
