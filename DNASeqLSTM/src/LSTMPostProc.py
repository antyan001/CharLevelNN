
import heapq
import pandas as pd
import numpy as np
import random
import keras.models as kmodels

class LSTM_pred(kmodels.Sequential):
    def __init__(self, model):
        super(LSTM_pred, self).__init__()
        self.model = model

    @staticmethod
    def _prepare_input(text, SEQUENCE_LENGTH, chars, char_indices):
        x = np.zeros((1, 2*SEQUENCE_LENGTH, len(chars)))
        for t, char in enumerate(text):
            x[0, t, char_indices[char]] = 1.

        return x

    @staticmethod
    def _sample(preds, top_n=3):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        return heapq.nlargest(top_n, range(len(preds)), preds.take)

    def predict_completion_one(self, text, SEQUENCE_LENGTH,
                               chars, char_indices, indices_char):
        original_text = text
        generated = text
        completion = ''
        while True:
            x = LSTM_pred._prepare_input(text, SEQUENCE_LENGTH,
                                         chars, char_indices)
            preds = self.model.predict(x, verbose=0)[0]
            next_index = LSTM_pred._sample(preds, top_n=1)[0]
            next_char = indices_char[next_index]
            return next_char

    def predict_completion_feed(self, text, SEQUENCE_LENGTH,
                                chars, char_indices, indices_char):
        original_text = text
        generated = text
        completion = ''
        while True:
            x = LSTM_pred._prepare_input(text, SEQUENCE_LENGTH,
                                         chars, char_indices)
            preds = self.model.predict(x, verbose=0)[0]
            next_index = LSTM_pred._sample(preds, top_n=1)[0]
            next_char = indices_char[next_index]
            text = text[1:] + next_char
            completion += next_char
            if len(original_text + completion) + 2 > len(original_text):
    #             and next_char == ' ':
                return completion

    def predict_completions(self, text, SEQUENCE_LENGTH,
                            chars, char_indices, indices_char,
                            n=3):
        x = LSTM_pred._prepare_input(text, SEQUENCE_LENGTH,
                                     chars, char_indices)
        preds = self.model.predict(x, verbose=0)[0]
        next_indices = LSTM_pred._sample(preds, n)
        return [indices_char[idx] + self.predict_completion_feed( text[1:] + indices_char[idx],
                                                                SEQUENCE_LENGTH,
                                                                chars,
                                                                char_indices,
                                                                indices_char) for idx in next_indices]
