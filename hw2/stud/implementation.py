import logging
import numpy as np
from typing import List, Tuple, Dict

from model import Model
import random

import json
from collections import defaultdict

# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')
from collections import namedtuple

from typing import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pytorch_lightning as pl

import os

import pandas as pd
import numpy as np

import random
import re

from transformers import DistilBertModel, DistilBertConfig
from datasets import load_metric

from transformers import DistilBertTokenizerFast, DistilBertModel, DistilBertForTokenClassification, TrainingArguments, Trainer, DistilBertForSequenceClassification
from tokenizers.processors import TemplateProcessing

# for cute iteration bars (during training etc.)
from tqdm.auto import tqdm

SEED = 1234

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUGGING = False


def build_model_b(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements aspect sentiment analysis of the ABSA pipeline.
            b: Aspect sentiment analysis.
    """
    # return RandomBaseline()
    return StudentModel(mode='b')

def build_model_ab(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline.
            a: Aspect identification.
            b: Aspect sentiment analysis.

    """
    # return RandomBaseline(mode='ab')
    return StudentModel(mode='ab')

def build_model_cd(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements both aspect identification and sentiment analysis of the ABSA pipeline 
        as well as Category identification and sentiment analysis.
            c: Category identification.
            d: Category sentiment analysis.
    """
    # return RandomBaseline(mode='cd')
    # raise NotImplementedError
    return StudentModel(mode='cd')

class RandomBaseline(Model):

    options_sent = [
        ('positive', 793+1794),
        ('negative', 701+638),
        ('neutral',  365+507),
        ('conflict', 39+72),
    ]

    options = [
        (0, 452),
        (1, 1597),
        (2, 821),
        (3, 524),
    ]

    options_cat_n = [
        (1, 2027),
        (2, 402),
        (3, 65),
        (4, 6),
    ]

    options_sent_cat = [
        ('positive', 1801),
        ('negative', 672),
        ('neutral',  411),
        ('conflict', 164),
    ]

    options_cat = [
        ("anecdotes/miscellaneous", 939),
        ("price", 268),
        ("food", 1008),
        ("ambience", 355),
        ("service", 248),
    ]

    def __init__(self, mode = 'b'):

        self._options_sent = [option[0] for option in self.options_sent]
        self._weights_sent = np.array([option[1] for option in self.options_sent])
        self._weights_sent = self._weights_sent / self._weights_sent.sum()

        if mode == 'ab':
            self._options = [option[0] for option in self.options]
            self._weights = np.array([option[1] for option in self.options])
            self._weights = self._weights / self._weights.sum()
        elif mode == 'cd':
            self._options_cat_n = [option[0] for option in self.options_cat_n]
            self._weights_cat_n = np.array([option[1] for option in self.options_cat_n])
            self._weights_cat_n = self._weights_cat_n / self._weights_cat_n.sum()

            self._options_sent_cat = [option[0] for option in self.options_sent_cat]
            self._weights_sent_cat = np.array([option[1] for option in self.options_sent_cat])
            self._weights_sent_cat = self._weights_sent_cat / self._weights_sent_cat.sum()

            self._options_cat = [option[0] for option in self.options_cat]
            self._weights_cat = np.array([option[1] for option in self.options_cat])
            self._weights_cat = self._weights_cat / self._weights_cat.sum()

        self.mode = mode

    def predict(self, samples: List[Dict]) -> List[Dict]:
        preds = []
        for sample in samples:
            pred_sample = {}
            words = None
            if self.mode == 'ab':
                n_preds = np.random.choice(self._options, 1, p=self._weights)[0]
                if n_preds > 0 and len(sample["text"].split(" ")) > n_preds:
                    words = random.sample(sample["text"].split(" "), n_preds)
                elif n_preds > 0:
                    words = sample["text"].split(" ")
            elif self.mode == 'b':
                if len(sample["targets"]) > 0:
                    words = [word[1] for word in sample["targets"]]
            if words:
                pred_sample["targets"] = [(word, str(np.random.choice(self._options_sent, 1, p=self._weights_sent)[0])) for word in words]
            else: 
                pred_sample["targets"] = []
            if self.mode == 'cd':
                n_preds = np.random.choice(self._options_cat_n, 1, p=self._weights_cat_n)[0]
                pred_sample["categories"] = []
                for i in range(n_preds):
                    category = str(np.random.choice(self._options_cat, 1, p=self._weights_cat)[0]) 
                    sentiment = str(np.random.choice(self._options_sent_cat, 1, p=self._weights_sent_cat)[0]) 
                    pred_sample["categories"].append((category, sentiment))
            preds.append(pred_sample)
        return preds

class BertDataset(Dataset):
    def __init__(self, 
                 inputs:List[Dict], 
                 tokenizer: DistilBertTokenizerFast,
                 mode:str,
                 lowercase=False, 
                 debugging=DEBUGGING,
                 device=DEVICE):
        
        self.inputs = inputs
        self.mode = mode
        self.lowercase = lowercase
        self.debugging = debugging
        self.device = DEVICE

        self.punctuation =  '[,.:?!]'

        self.tokenizer = tokenizer

        self.tags = ['O', 'B']

        self.unique_tags = {'O', 'B'}
        self.tag2id =  {'O': 0, 'B': 1}
        self.id2tag = {id: tag for tag, id in self.tag2id.items()}

        self.unique_polarity = {'positive', 'negative', 'neutral', 'conflict'}
        self.polarity2id = {'positive': 0, 'negative': 1, 'neutral': 2, 'conflict': 3}
        self.id2polarity = {id: polarity for polarity, id in self.polarity2id.items()}

        self.unique_categories = {'ambience', 'anecdotes/miscellaneous', 'food', 'price', 'service'}
        self.category2id = {'ambience': 0, 'anecdotes/miscellaneous': 1, 'food': 2, 'price': 3, 'service': 4}
        self.id2category = {id: category for category, id in self.category2id.items()}

        self.encoded_input= None

        self.sentences = self.tokenize_sentece(inputs)

    def regex_split(self, text) -> str:
        return re.findall(r"[\w']+|"+self.punctuation, text)

    def tokenize_sentece(self, inputs: List[Dict]) -> List[List[str]]:
        sentences = {"text": [], "targets": [], "categories": [], "full_sentence":[]}
        for input in inputs:
            text = input['text']
            text_ = input['text']
            
            if self.lowercase:
                text = text.lower()

            # Regex splitting (keep the punctuation)
            tokenized_sentence = self.regex_split(text)

            if self.mode == 'b':
                targets = input['targets']
                for target in targets:
                    target_words = self.regex_split(target[1])
                    sentences['text'].append(tokenized_sentence)
                    sentences['targets'].append(target_words)
                    sentences['full_sentence'].append(text_)
                if not targets: #If there are no targets
                    # Append to the list of sentences
                    sentences['text'].append(tokenized_sentence)
                    sentences['targets'].append([''])
                    sentences['full_sentence'].append(text_)
            elif self.mode == 'd':
                categories = input['categories']
                for category in categories:
                    cat = [category[0]]
                    sentences['text'].append(tokenized_sentence)
                    sentences['categories'].append(cat)
                    sentences['full_sentence'].append(text_)             
                if not categories:
                    sentences['text'].append(tokenized_sentence)
                    sentences['categories'].append(cat)
                    sentences['full_sentence'].append(text_)
            elif self.mode == 'ab' or  self.mode == 'cd':
                # Append to the list of sentences
                sentences['text'].append(tokenized_sentence)
                sentences['full_sentence'].append(text_)

        return sentences

    def __len__(self):
        return len(self.sentences['text'])

    def __getitem__(self, idx):
        if self.encoded_input is None:
            raise RuntimeError("""Trying to retrieve elements but index_dataset
            has not been invoked yet! Be sure to invoce index_dataset on this object
            before trying to retrieve elements. In case you want to retrieve raw
            elements, use the method get_raw_element(idx)""")

        item = {key: torch.tensor(val[idx]) for key, val in self.encoded_input.items()}
        return item

    def get_raw_sentence(self, idx):
        if self.mode == 'ab' or self.mode == 'cd': 
            return self.sentences['text'][idx], self.sentences['full_sentence'][idx]
        elif self.mode == 'b':
            return self.sentences['text'][idx], self.sentences['targets'][idx], self.sentences['full_sentence'][idx]
        elif self.mode == 'd':
            return self.sentences['text'][idx], self.sentences['categories'][idx], self.sentences['full_sentence'][idx]

    def encode_dataset(self) -> None:
        if self.mode == 'ab' or self.mode == 'cd':
            self.encoded_input  = self.tokenizer(self.sentences['text'], is_split_into_words=True, padding=True, truncation=True, return_tensors='pt')
        elif self.mode == 'b':
            self.encoded_input  = self.tokenizer(self.sentences['text'], self.sentences['targets'], is_split_into_words=True, padding=True, truncation=True, return_tensors='pt')
        elif self.mode == 'd':
            self.encoded_input  = self.tokenizer(self.sentences['text'], self.sentences['categories'], is_split_into_words=True, padding=True, truncation=True, return_tensors='pt')

    def decode_sentence_tags(self, tokenized_sentece, tags) -> List[Tuple[str, str]]:
        reconstructed_sentence = []
        r_tags = []
        for index, token in enumerate(tokenized_sentece):
            decoded_token = self.tokenizer.decode(token) 
            if decoded_token.startswith("##"):
                if reconstructed_sentence:
                    reconstructed_sentence[-1] = f"{reconstructed_sentence[-1]}{decoded_token[2:]}"
            else:
                reconstructed_sentence.append(decoded_token)
                r_tags.append(tags[index])

        return reconstructed_sentence, r_tags


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode

        self.model_a = DistilBertForTokenClassification.from_pretrained('./model/a/', local_files_only=True, num_labels=2).to(DEVICE)
        self.model_a.eval() # Bert is set to eval mode by default
        self.tokenizer_a = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

        self.treshold_conflict_b = 0.05
        self.model_b = DistilBertForSequenceClassification.from_pretrained('./model/b/', local_files_only=True, num_labels=4).to(DEVICE)
        self.model_b.eval() # Bert is set to eval mode by default
        self.tokenizer_b = DistilBertTokenizerFast.from_pretrained('./model/tokenizer/b/')
        self.tokenizer_b.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )

        self.model_c = DistilBertForSequenceClassification.from_pretrained('./model/c/', num_labels=5).to(DEVICE)
        self.model_c.eval()
        self.tokenizer_c = DistilBertTokenizerFast.from_pretrained('./model/tokenizer/c/') # Same as tokenizer a

        self.treshold_conflict_d = 0.05
        self.model_d = DistilBertForSequenceClassification.from_pretrained('./model/d/', local_files_only=True, num_labels=4).to(DEVICE)
        self.model_d.eval() # Bert is set to eval mode by default
        self.tokenizer_d = DistilBertTokenizerFast.from_pretrained('./model/tokenizer/d/')
        self.tokenizer_d.post_processor = TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", 1),
                ("[SEP]", 2),
            ],
        )

    def predict(self, samples: List[Dict]) -> List[Dict]:
        '''
        --> !!! STUDENT: implement here your predict function !!! <--
        Args:
            - If you are doing model_b (ie. aspect sentiment analysis):
                sentence: a dictionary that represents an input sentence as well as the target words (aspects), for example:
                    [
                        {
                            "text": "I love their pasta but I hate their Ananas Pizza.",
                            "targets": [[13, 17], "pasta"], [[36, 47], "Ananas Pizza"]]
                        },
                        {
                            "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                            "targets": [[4, 9], "people", [[36, 40], "taste"]]
                        }
                    ]
            - If you are doing model_ab or model_cd:
                sentence: a dictionary that represents an input sentence, for example:
                    [
                        {
                            "text": "I love their pasta but I hate their Ananas Pizza."
                        },
                        {
                            "text": "The people there is so kind and the taste is exceptional, I'll come back for sure!"
                        }
                    ]
        Returns:
            A List of dictionaries with your predictions:
                - If you are doing target word identification + target polarity classification:
                    [
                        {
                            "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")] # A list having a tuple for each target word
                        },
                        {
                            "targets": [("people", "positive"), ("taste", "positive")] # A list having a tuple for each target word
                        }
                    ]
                - If you are doing target word identification + target polarity classification + aspect category identification + aspect category polarity classification:
                    [
                        {
                            "targets": [("pasta", "positive"), ("Ananas Pizza", "negative")], # A list having a tuple for each target word
                            "categories": [("food", "conflict")]
                        },
                        {
                            "targets": [("people", "positive"), ("taste", "positive")], # A list having a tuple for each target word
                            "categories": [("service", "positive"), ("food", "positive")]
                        }
                    ]
        '''

        if self.mode == 'ab':
            testset = BertDataset(samples, self.tokenizer_a, self.mode, lowercase=False)
            testset.encode_dataset()
            return self.predict_ab(testset)
        elif self.mode == 'b':
            testset = BertDataset(samples, self.tokenizer_b, self.mode, lowercase=False)
            testset.encode_dataset()
            return self.predict_b(testset)
        elif self.mode == 'cd':
            testset = BertDataset(samples, self.tokenizer_c, self.mode, lowercase=False)
            testset.encode_dataset()
            return self.predict_cd(testset)


    def predict_b(self, dataset: BertDataset) -> List[Dict]:
        outputs = []
        last_sentence = ''
        # Iterate over sentences
        for idx in range(len(dataset)):     
            # Take sentence and targets            
            text, target, full_sentence = dataset.get_raw_sentence(idx)
            # If there are no targets the output is an empty list
            if len(target) == 1 and target[0] == '':
                outputs.append({"targets":[]})
            else:
                # Tokenize the sentence
                inputs = self.tokenizer_b(text, target, is_split_into_words=True, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                # Take the prediction from the model
                with torch.no_grad():
                    output = self.model_b(**inputs).logits

                sorted_tensor = torch.sort(output, descending=True)[0][0]
                if abs(sorted_tensor[0]-sorted_tensor[1]) < self.treshold_conflict_b:
                    prediction = torch.tensor(dataset.polarity2id['conflict'])
                else:
                    prediction = torch.argmax(output, dim=1)[0]

                predicted_polarity = dataset.id2polarity[prediction.item()]

                reconstructed_target = ''
                for elem in target:
                    space = ' ' if elem not in dataset.punctuation and reconstructed_target else ''
                    reconstructed_target += f"{space}{elem}"

                current_sentece = full_sentence

                if current_sentece == last_sentence:
                    outputs[-1]['targets'].append((reconstructed_target, predicted_polarity))
                else:
                    outputs.append({"targets": [(reconstructed_target, predicted_polarity)]})

                last_sentence = current_sentece
        return outputs

    def predict_ab(self, dataset: BertDataset) -> List[Dict]:
        skip_tokens = ['[CLS]', '[SEP]']

        samples = []
        # Iterate over sentences
        for idx in range(len(dataset)):
            # Get te sentence
            sentence, full_sentence = dataset.get_raw_sentence(idx)
            # Tokenize
            inputs = self.tokenizer_a(sentence, is_split_into_words=True, return_tensors='pt').to(DEVICE)
            # Take the prediction from the model
            with torch.no_grad():
                output = self.model_a(**inputs).logits
            prediction = torch.argmax(output, dim=2)[0]
            # Decode the predictions
            r, r_tags = dataset.decode_sentence_tags(inputs['input_ids'][0], prediction)

            concat_aspect_term = False
            aspect_terms = []
            # Iterate over token in sentence
            for token, predicted_tag in zip(r, r_tags):
                if token not in skip_tokens:
                    if predicted_tag == dataset.tag2id['B']:
                        if concat_aspect_term:
                            space = ' ' if token not in dataset.punctuation else ''
                            aspect_terms[-1] = aspect_terms[-1] + f"{space}{token}"
                        else:
                            aspect_terms.append(token)
                        concat_aspect_term = True
                    else:
                        concat_aspect_term = False

            targets = [[[], aspect_term] for aspect_term in aspect_terms]
            samples.append({"targets": targets, "text": full_sentence})

        # Predict the polarity of the found aspect term
        dataset_b = BertDataset(samples, self.tokenizer_b, mode='b', lowercase=False)
        dataset_b.encode_dataset()
        return self.predict_b(dataset_b)

    def predict_d(self, dataset: BertDataset) -> List[Dict]:
        outputs = []
        last_sentence = ''
        # Iterate over sentences
        for idx in range(len(dataset)):     
            # Take sentence and targets            
            text, category, full_sentence = dataset.get_raw_sentence(idx)
            # If there are no targets the output is an empty list
            if len(category) == 1 and category[0] == '':
                outputs.append({"categories":[]})
            else:
                # Tokenize the sentence
                inputs = self.tokenizer_d(text, category, is_split_into_words=True, padding=True, truncation=True, return_tensors='pt').to(DEVICE)
                # Take the prediction from the model
                with torch.no_grad():
                    output = self.model_d(**inputs).logits
                prediction = torch.argmax(output, dim=1)[0]

                sorted_tensor = torch.sort(output, descending=True)[0][0]
                if abs(sorted_tensor[0]-sorted_tensor[1]) < self.treshold_conflict_d:
                    # prediction = torch.tensor(dataset.polarity2id['conflict'])
                    predicted_polarity = 'conflict'
                else:
                    predicted_polarity = dataset.id2polarity[prediction.item()]

                # predicted_polarity = dataset.id2polarity[prediction.item()]

                current_sentece = full_sentence

                if current_sentece == last_sentence:
                    outputs[-1]['categories'].append((*category, predicted_polarity))
                else:
                    outputs.append({"categories": [(*category, predicted_polarity)]})

                last_sentence = current_sentece
        return outputs

    def predict_cd(self, dataset: BertDataset) -> List[Dict]:
        samples = []
        last_sentence = ''
        for idx in range(len(dataset)): 
            text, full_sentence = dataset.get_raw_sentence(idx)

            inputs = self.tokenizer_c(text, is_split_into_words=True, return_tensors='pt').to(DEVICE)
            output = self.model_c(**inputs).logits
            with torch.no_grad():
                prediction = torch.argmax(output, dim=1)[0]

            prediction = dataset.id2category[prediction.item()]

            samples.append({"categories": [[prediction]], "text": full_sentence})

        # Predict the polarity of the found aspect term
        dataset_d = BertDataset(samples, self.tokenizer_d, mode='d', lowercase=False)
        dataset_d.encode_dataset()
        return self.predict_d(dataset_d)
            

        



