from utils.utils import calculate_accracy
import numpy as np
import nltk
import tensorflow as tf

def display_sentences(batch, decoded_sentence, n, token_embedder, output_file):
    batch_size = len(batch["sentence"])
    indexes = np.random.choice(batch_size, n, replace=False)
    for i in indexes:
        l = len(batch["sentence"][i])
        s = token_embedder.add_unk_plaintext(batch["sentence"][i])
        print("Source: " + " ".join(s))
        sentence = token_embedder.gen_sentence_from_indexes_no_repeat(decoded_sentence[i])[:l]
        print("Predicted: " + " ".join(sentence))

        output_file.write("Source: " + " ".join(s) + "\n")
        output_file.write("Predicted: " + " ".join(sentence) + "\n")

def write_all_sentences(batch, decoded_sentence, token_embedder, output_file):
    batch_size = len(batch["sentence"])
    for i in range(batch_size):
        l = len(batch["sentence"][i])
        s = token_embedder.add_unk_plaintext(batch["sentence"][i])
        sentence = token_embedder.gen_sentence_from_indexes_no_repeat(decoded_sentence[i])[:l]

        output_file.write("Source: " + " ".join(s) + "\n")
        output_file.write("Predicted: " + " ".join(sentence) + "\n")

def interpolate_sentences(batch1, batch2, decoded_sentence, token_embedder, output_file):
    batch_size = len(batch1["sentence"])
    for i in range(batch_size):
        l = max(len(batch1["sentence"][i]),len(batch2["sentence"][i]))
        s = token_embedder.add_unk_plaintext(batch1["sentence"][i])
        s2 = token_embedder.add_unk_plaintext(batch2["sentence"][i])
        sentence = token_embedder.gen_sentence_from_indexes_no_repeat(decoded_sentence[i])[:l]

        output_file.write("Sentence1: " + " ".join(s) + "\n")
        output_file.write("Sentence2: " + " ".join(s2) + "\n")
        output_file.write("Predicted: " + " ".join(sentence) + "\n")

def display_interact_sentences(batch, decoded_sentence, input_size, token_embedder, output_file):
    for i in range(input_size):
        l = len(batch["sentence"][i])
        s = token_embedder.add_unk_plaintext(batch["sentence"][i])
        print("Source: " + " ".join(s))
        sentence = token_embedder.gen_sentence_from_indexes_no_repeat(decoded_sentence[i])[:l]
        print("Predicted: " + " ".join(sentence))
        output_file.write("Source: " + " ".join(s) + "\n")
        output_file.write("Predicted: " + " ".join(sentence) + "\n")

def calc_accuracy(batch, decoded_sentence, token_embedder, key):
    predicted_sentences = []
    target_sentences = []
    batch_size = len(batch[key])
    for i in range(batch_size):
        predicted_sentences.append(token_embedder.gen_sentence_from_indexes(decoded_sentence[i]))
        target_sentences.append(" ".join(batch[key][i]))

    acc = calculate_accracy(target_sentences, predicted_sentences)
    return acc


def calc_bleu(batch, decoded_sentence, token_embedder, key):
    bleu_total = 0.0
    target_sentences = batch[key]
    batch_size = len(batch[key])
    for i in range(batch_size):
        hyp = token_embedder.gen_sentence_from_indexes_no_repeat(decoded_sentence[i])
        ref = [target_sentences[i]]
        try:
            bleu_total += nltk.translate.bleu_score.sentence_bleu(ref, hyp)
        except:
            bleu_total = bleu_total

    bleu = bleu_total / batch_size
    return bleu

def write_scaler_summaries(train_bleu, test_bleu, train_word_acc, test_word_acc, summary_writer_train, summary_writer_validation, step):
    train_word_acc_summary = tf.Summary(value=[tf.Summary.Value(tag="word_accuracy", simple_value=train_word_acc)])
    train_bleu_summary = tf.Summary(value=[tf.Summary.Value(tag="bleu", simple_value=train_bleu)])

    summary_writer_train.add_summary(train_word_acc_summary, step)
    summary_writer_train.add_summary(train_bleu_summary, step)

    test_word_acc_summary = tf.Summary(
        value=[tf.Summary.Value(tag="word_accuracy", simple_value=test_word_acc)])
    test_bleu_summary = tf.Summary(value=[tf.Summary.Value(tag="bleu", simple_value=test_bleu)])

    summary_writer_validation.add_summary(test_word_acc_summary, step)
    summary_writer_validation.add_summary(test_bleu_summary, step)

def print_and_write_training_info(output_file, loss, word_acc, bleu):
    print("Loss: " + str(loss))
    print("Word Accuracy: " + str(word_acc))
    print("Bleu Score: " + str(bleu))

    output_file.write("Loss: " + str(loss) + "\n")
    output_file.write("Word Accuracy: " + str(word_acc) + "\n")
    output_file.write("Bleu Score: " + str(bleu) + "\n")

def jaccard(target_sentences, sentence_tensors, token_embedder):
    total_distance = 0
    batch_size = len(target_sentences)
    for i in range(batch_size):
        words_set1 = set(target_sentences[i])
        words_set2 = set(token_embedder.gen_sentence_from_indexes_no_repeat(sentence_tensors[i]))
        intersection = len(words_set1.intersection(words_set2))
        union = len(words_set1.union(words_set2))
        total_distance += 1 - float(intersection) / float(union)
    return total_distance/batch_size

def levenshtein(target_sentences, sentence_tensors, token_embedder):
    total_distance = 0
    batch_size = len(target_sentences)
    for i in range(batch_size):
        seq1 = target_sentences[i]
        seq2 = token_embedder.gen_sentence_from_indexes_no_repeat(sentence_tensors[i])
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros ((size_x, size_y))
        for x in range(size_x):
            matrix [x, 0] = x
        for y in range(size_y):
            matrix [0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x-1] == seq2[y-1]:
                    matrix [x,y] = min(
                        matrix[x-1, y] + 1,
                        matrix[x-1, y-1],
                        matrix[x, y-1] + 1
                    )
                else:
                    matrix [x,y] = min(
                        matrix[x-1,y] + 1,
                        matrix[x-1,y-1] + 1,
                        matrix[x,y-1] + 1
                    )
        total_distance += matrix[size_x - 1, size_y - 1]
    return total_distance / batch_size

