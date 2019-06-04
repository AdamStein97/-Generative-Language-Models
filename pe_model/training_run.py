import time

from pe_model.editor import Editor
from utils.token_embedder import TokenEmbedder
import pe_model.utils_pe as utils
from utils.training_utils import print_and_write_training_info,calc_accuracy, calc_bleu, write_scaler_summaries, levenshtein, jaccard


from utils.utils import greedy_decode

import tensorflow as tf
import random
import numpy as np


class TrainingRun():
    def __init__(self, config, debug, output_file_name):
        self.config = config
        self.output_file_name = self.config["log_dir"] + output_file_name
        self.output_file = open(self.output_file_name, "w")

        if debug:
            tf.set_random_seed(1)
            random.seed(1)
            np.random.seed(1)


        self.token_embedder = TokenEmbedder(self.config["word_embedding_directory"], self.config['max_sentence_length'])
        self.train_set = utils.examples_from_file("train.tsv", self.config["dataset_directroy"], self.token_embedder,self.config['ignore_free_set'])
        self.validation_set = utils.examples_from_file("valid.tsv", self.config["dataset_directroy"], self.token_embedder, self.config['ignore_free_set'])
        self.batch_size = config["batch_size"]

        self._init_input_tensors()
        self._build_architecture()
        self._init_post_run_tensors()
        self._init_optimizers()
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        self._init_tensorboard_summaries()

    def _init_input_tensors(self):
        self.source = tf.placeholder(tf.float32, [self.config["batch_size"], self.config['max_sentence_length'], self.config["word_dim"]])
        self.target_ohv = tf.placeholder(tf.float32, [self.config["batch_size"], self.config['max_sentence_length']])
        self.insert_words = tf.placeholder(tf.float32,
                                      [self.config["batch_size"], 5, self.config["word_dim"]])
        self.delete_words = tf.placeholder(tf.float32,
                                      [self.config["batch_size"], 5, self.config["word_dim"]])
        self.seq_lengths = tf.placeholder(tf.int32, [self.config["batch_size"]])
        self.target_masks = tf.placeholder(tf.float32, [self.config["batch_size"], self.config['max_sentence_length']])
        self.step = tf.placeholder(tf.float32)

    def _build_architecture(self):
        self.editor = Editor(self.config["hidden_dim"], self.config["norm_eps"], self.config["norm_max"], self.config["kappa"],
                        self.config["agenda_dim"], self.config["batch_size"], self.token_embedder, self.config["encoder_layers"],
                        self.config["decoder_layers"], self.config["edit_dim"], self.config["agenda_augment"], self.config["attention_dim"],
                             self.config["bidirectional"], self.config["attention"], self.config["decoder_dropout_perc"],
                             self.config["dist_name"], self.config["hold_state"])

    def _init_optimizers(self):
        variables_gp = [v for v in tf.trainable_variables() if ("q_model" not in v.name)]
        variables_gq = [v for v in tf.trainable_variables() if ("q_model" in v.name)]
        #self.train_gp_op = tf.train.AdamOptimizer(self.config["learning_rate"]).minimize(self.loss)
        self.train_gq_op = tf.train.AdamOptimizer(self.config["learning_rate"]).minimize(self.loss)

    def _init_post_run_tensors(self):
        self.y_prime = self.editor.foward_train(self.source, self.insert_words, self.delete_words, self.seq_lengths)
        self.greedy_decode_sentence = greedy_decode(self.batch_size, self.y_prime)
        self.loss, self.kl_val, self.kl_weight, self.var_param, self.logp = self.editor.loss(self.y_prime, self.target_ohv, self.target_masks, self.step)


    def _predict_sentences(self, batch, decoded_sentence, n):
        batch_size = self.config["batch_size"]
        indexes = np.random.choice(batch_size, n, replace=False)
        for i in indexes:
            l = len(batch["targets_plaintext"][i])
            print("Source: " + " ".join(batch["sources_plaintext"][i]))
            print("Target: " + " ".join(batch["targets_plaintext"][i]))
            sentence = self.token_embedder.gen_sentence_from_indexes_no_repeat(decoded_sentence[i])[:l]
            print("Predicted: " + " ".join(sentence))

            self.output_file.write("Source: " + " ".join(batch["sources_plaintext"][i]) + "\n")
            self.output_file.write("Target: " + " ".join(batch["targets_plaintext"][i]) + "\n")
            self.output_file.write("Predicted: " + " ".join(sentence) + "\n")

    def _init_tensorboard_summaries(self):
        kl_value_summary = tf.summary.scalar('KL_Value', self.kl_val)
        kl_weight_summary = tf.summary.scalar('KL_Weight', self.kl_weight)
        loss_summary = tf.summary.scalar('Loss', self.loss)
        var_param = tf.summary.scalar('Var', self.var_param)
        reconstruction_loss = tf.summary.scalar('Reconstruction loss', self.logp)
        self.run_summary = tf.summary.merge(
            [loss_summary, kl_value_summary, kl_weight_summary, var_param, reconstruction_loss])


    def train(self, restore_state):
            with tf.Session() as sess:
                if self.config["log"] == 1:
                    summary_writer_train = tf.summary.FileWriter(self.config["log_dir"] + 'train_log', sess.graph,
                                                                 flush_secs=5)
                    summary_writer_validation = tf.summary.FileWriter(self.config["log_dir"] + 'validation_log', sess.graph,
                                                                      flush_secs=5)
                sess.run(tf.global_variables_initializer())
                if restore_state:
                    model_checkpoint = self.config["model_save_directory"] + self.config["model_restore_name"]
                    self.saver.restore(sess, model_checkpoint)

                start_time = time.time()
                for step in range(self.config["training_steps"]):
                    total_time = time.time() - start_time
                    if (total_time / 60 < self.config["max_train_mins"]):
                        batch_data = utils.create_batch(self.train_set, self.config["batch_size"], self.token_embedder, step)

                        t0 = time.time()

                        train_loss_gq, _, sentences, summary_str, kl, kl_weight,var, rc = sess.run([self.loss, self.train_gq_op, self.greedy_decode_sentence, self.run_summary
                                                                             , self.kl_val,self.kl_weight, self.var_param, self.logp],
                                                               feed_dict={self.source: batch_data['sources'],
                                                                         self.insert_words: batch_data[
                                                                             'insert_words'],
                                                                         self.delete_words: batch_data[
                                                                             'delete_words'],
                                                                         self.target_ohv: batch_data[
                                                                             'target_ohv'],
                                                                          self.seq_lengths: batch_data["seq_lengths"],
                                                                          self.target_masks: batch_data["mask"],
                                                                          self.step: step})

                        t1 = time.time()

                        if self.config["log"] == 1:
                            summary_writer_train.add_summary(summary_str, step)

                        if (step + 1) % self.config["save_frequency"] == 0:
                            checkpoint_path = self.config["model_save_directory"] + 'trained_model.ckpt'
                            self.saver.save(sess, checkpoint_path, global_step=step)

                        if (step + 1) % self.config["big_evaluation"] == 0:
                            print("---------BIG EVALUATION---------")
                            print("Step: " + str(step))
                            self.output_file.write("---------BIG EVALUATION---------\n")
                            self.output_file.write("Step: " + str(step) + "\n")
                            self.output_file.write("Batches Over: " + str(self.config["batches_big_evaluation"]) + "\n")
                            train_loss_total = 0
                            test_loss_total = 0
                            train_reconstruct_total = 0
                            test_reconstruct_total = 0
                            total_kl = 0
                            train_bleu_score_total = 0
                            test_bleu_score_total = 0
                            train_word_acc_total = 0
                            test_word_acc_total = 0
                            test_jaccard_total = 0
                            test_levenshtein_total = 0
                            eval_time = time.time()
                            for i in range(self.config["batches_big_evaluation"]):
                                batch_data = utils.create_batch(self.train_set, self.config["batch_size"],
                                                                self.token_embedder, step + i)
                                test_batch = utils.create_batch(self.validation_set, self.config["batch_size"],
                                                                self.token_embedder, step + i)

                                train_loss, kl, sentence_train, trc = sess.run(
                                    [self.loss, self.kl_val, self.greedy_decode_sentence, self.logp],
                                    feed_dict={self.source: batch_data['sources'],
                                               self.insert_words: batch_data[
                                                   'insert_words'],
                                               self.delete_words: batch_data[
                                                   'delete_words'],
                                               self.target_ohv: batch_data[
                                                   'target_ohv'],
                                               self.seq_lengths: batch_data["seq_lengths"],
                                               self.target_masks: batch_data["mask"],
                                               self.step: step})
                                word_acc_train = calc_accuracy(batch_data, sentence_train, self.token_embedder,
                                                               "sources_plaintext")
                                bleu_train = calc_bleu(batch_data, sentence_train, self.token_embedder, "sources_plaintext")

                                test_loss, test_reconstruction_loss, sentence_test = sess.run(
                                    [self.loss, self.logp, self.greedy_decode_sentence],
                                    feed_dict={self.source: test_batch['sources'],
                                               self.insert_words: test_batch[
                                                   'insert_words'],
                                               self.delete_words: test_batch[
                                                   'delete_words'],
                                               self.target_ohv: test_batch[
                                                   'target_ohv'],
                                               self.seq_lengths: test_batch["seq_lengths"],
                                               self.target_masks: test_batch["mask"],
                                               self.step: step})
                                word_acc_test = calc_accuracy(test_batch, sentence_test, self.token_embedder,
                                                              "sources_plaintext")
                                bleu_test = calc_bleu(test_batch, sentence_test, self.token_embedder, "sources_plaintext")

                                jac_d = jaccard(test_batch["sources_plaintext"], sentence_test, self.token_embedder)
                                l_d = levenshtein(test_batch["sources_plaintext"], sentence_test, self.token_embedder)

                                train_loss_total += train_loss
                                test_loss_total += test_loss
                                test_reconstruct_total += test_reconstruction_loss
                                total_kl += kl
                                train_word_acc_total += word_acc_train
                                test_word_acc_total += word_acc_test
                                train_bleu_score_total += bleu_train
                                test_bleu_score_total += bleu_test
                                test_jaccard_total += jac_d
                                test_levenshtein_total += l_d
                                train_reconstruct_total += trc
                            print("Evaluation Time: " + str(time.time() - eval_time))
                            average_train_loss = train_loss_total / self.config["batches_big_evaluation"]
                            average_test_loss = test_loss_total / self.config["batches_big_evaluation"]
                            average_reconstruction_loss = test_reconstruct_total / self.config["batches_big_evaluation"]
                            average_reconstruction_loss_train = train_reconstruct_total / self.config["batches_big_evaluation"]
                            average_kl = total_kl / self.config["batches_big_evaluation"]
                            test_perplexity = 2 ** (average_reconstruction_loss / np.log(2))
                            average_acc_train = train_word_acc_total / self.config["batches_big_evaluation"]
                            average_acc_test = test_word_acc_total / self.config["batches_big_evaluation"]
                            average_bleu_train = train_bleu_score_total / self.config["batches_big_evaluation"]
                            average_bleu_test = test_bleu_score_total / self.config["batches_big_evaluation"]
                            average_jac = test_jaccard_total / self.config["batches_big_evaluation"]
                            average_lev = test_levenshtein_total / self.config["batches_big_evaluation"]

                            print("Training Loss: " + str(average_train_loss))
                            print("Testing Loss: " + str(average_test_loss))
                            print("Reconstruction Train Loss: " + str(average_reconstruction_loss_train))
                            print("Reconstruction Test Loss: " + str(average_reconstruction_loss))
                            print("Testing Perplexity: " + str(test_perplexity))
                            print("KL Divergence: " + str(average_kl))
                            print("Training Word Accuracy: " + str(average_acc_train))
                            print("Testing Word Accuracy: " + str(average_acc_test))
                            print("Training BLEU Score: " + str(average_bleu_train))
                            print("Testing BLEU Score: " + str(average_bleu_test))

                            self.output_file.write("Evaluation Time: " + str(time.time() - eval_time) + "\n")
                            self.output_file.write("Training Loss: " + str(average_train_loss) + "\n")
                            self.output_file.write("Testing Loss: " + str(average_test_loss) + "\n")
                            self.output_file.write(
                                "Reconstruction Test Loss: " + str(average_reconstruction_loss) + "\n")
                            self.output_file.write(
                                "Reconstruction Train Loss: " + str(average_reconstruction_loss_train) + "\n")
                            self.output_file.write("Testing Perplexity: " + str(test_perplexity) + "\n")
                            self.output_file.write("KL Divergence: " + str(average_kl) + "\n")
                            self.output_file.write("Training Word Accuracy: " + str(average_acc_train) + "\n")
                            self.output_file.write("Testing Word Accuracy: " + str(average_acc_test) + "\n")
                            self.output_file.write("Training BLEU Score: " + str(average_bleu_train) + "\n")
                            self.output_file.write("Testing BLEU Score: " + str(average_bleu_test) + "\n")

                            if self.config["log"] == 1:
                                big_train_loss_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="Big Train Loss", simple_value=average_train_loss)])
                                big_test_loss_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="Big Test Loss", simple_value=average_test_loss)])
                                reconstruction_loss_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="Big Reconstruction Loss",
                                                            simple_value=average_reconstruction_loss)])
                                perplex_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="Perplexity", simple_value=test_perplexity)])
                                kl_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="Big KL Divergence", simple_value=average_kl)])
                                train_acc_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="Big Word Accuracy Train",
                                                            simple_value=average_acc_train)])
                                test_acc_summary = tf.Summary(
                                    value=[
                                        tf.Summary.Value(tag="Big Word Accuracy Test", simple_value=average_acc_test)])
                                train_bleu_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="Big Bleu Train", simple_value=average_bleu_train)])
                                test_bleu_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="Big Bleu Test", simple_value=average_bleu_test)])
                                jac_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="Big Jaccard Test", simple_value=average_jac)])
                                lev_summary = tf.Summary(
                                    value=[tf.Summary.Value(tag="Big Levenshtein Test", simple_value=average_lev)])

                                summary_writer_train.add_summary(big_train_loss_summary, step)
                                summary_writer_train.add_summary(kl_summary, step)
                                summary_writer_train.add_summary(train_acc_summary, step)
                                summary_writer_train.add_summary(train_bleu_summary, step)
                                summary_writer_validation.add_summary(big_test_loss_summary, step)
                                summary_writer_validation.add_summary(reconstruction_loss_summary, step)
                                summary_writer_validation.add_summary(perplex_summary, step)
                                summary_writer_validation.add_summary(test_acc_summary, step)
                                summary_writer_validation.add_summary(test_bleu_summary, step)
                                summary_writer_validation.add_summary(jac_summary, step)
                                summary_writer_validation.add_summary(lev_summary, step)

                        if step % self.config["eval_frequency"] == 0:
                            print("---------EVALUATION---------")
                            print("Step: " + str(step))
                            print("Iteration Time: " + str(t1 - t0))
                            print("Total Minutes Training: " + str(total_time / 60))
                            print("KL loss: " + str(kl))
                            print("Var Param: " + str(var))
                            print("Reconstruction Cost: " + str(rc))
                            self.output_file.write("---------EVALUATION---------\n")
                            self.output_file.write("Step: " + str(step) + "\n")
                            self.output_file.write("Total Minutes Training: " + str(total_time / 60) + "\n")
                            self.output_file.write("KL Loss: " + str(kl) + "\n")
                            self.output_file.write("KL Weight: " + str(kl_weight) + "\n")
                            self.output_file.write("Var Param: " + str(var) + "\n")
                            self.output_file.write("Reconstruction Loss: " + str(rc) + "\n")
                            self._predict_sentences(batch_data, sentences, 3)
                            train_word_acc = calc_accuracy(batch_data, sentences, self.token_embedder, "targets_plaintext")
                            train_bleu = calc_bleu(batch_data, sentences, self.token_embedder, "targets_plaintext")

                            print_and_write_training_info(self.output_file, train_loss_gq, train_word_acc, train_bleu)

                            print("---------VALIDATION---------")
                            self.output_file.write("---------VALIDATION---------\n")

                            test_batch = utils.create_batch(self.validation_set, self.config["batch_size"], self.token_embedder, step)
                            test_loss, sentences_test, summary_str_test = sess.run(
                                [self.loss, self.greedy_decode_sentence, self.run_summary],
                                feed_dict={self.source: test_batch['sources'],
                                           self.insert_words: test_batch[
                                               'insert_words'],
                                           self.delete_words: test_batch[
                                               'delete_words'],
                                           self.target_ohv: test_batch[
                                               'target_ohv'],
                                           self.seq_lengths: test_batch["seq_lengths"],
                                           self.target_masks: test_batch["mask"],
                                           self.step: step})
                            self._predict_sentences(test_batch, sentences_test, 3)

                            test_word_acc = calc_accuracy(test_batch, sentences, self.token_embedder, "targets_plaintext")
                            test_bleu = calc_bleu(test_batch, sentences, self.token_embedder, "targets_plaintext")
                            print_and_write_training_info(self.output_file, test_loss, test_word_acc, test_bleu)


                            if self.config["log"]:
                                summary_writer_validation.add_summary(summary_str_test, step)
                                write_scaler_summaries(train_bleu, test_bleu, train_word_acc, test_word_acc,
                                                       summary_writer_train, summary_writer_validation, step)

                            self.output_file.close()
                            self.output_file = open(self.output_file_name, "a")



