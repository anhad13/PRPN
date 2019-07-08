import os
import re
import pickle as cPickle
import copy

import numpy
import torch
import nltk
from nltk.corpus import ptb

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']
from nltk.corpus import BracketParseCorpusReader
corpus_root = r"/home/am8676/nltk_data/corpora/PTB/"
file_pattern = r".*/WSJ_.*\.MRG"
ptb = BracketParseCorpusReader(corpus_root, file_pattern)

file_ids = ptb.fileids()
train_file_ids = []
valid_file_ids = []
test_file_ids = []
rest_file_ids = []
for id in file_ids:
    if 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
        train_file_ids.append(id)
    elif 'WSJ/22/WSJ_2200.MRG' <= id <= 'WSJ/22/WSJ_2299.MRG':
        valid_file_ids.append(id)
    # elif 'WSJ/23/WSJ_2300.MRG' <= id <= 'WSJ/23/WSJ_2399.MRG':
    #     test_file_ids.append(id)
    # elif 'WSJ/00/WSJ_0000.MRG' <= id <= 'WSJ/01/WSJ_0199.MRG' or 'WSJ/24/WSJ_2400.MRG' <= id <= 'WSJ/24/WSJ_2499.MRG':
    #     rest_file_ids.append(id)
#train_file_ids = train_file_ids[:30]
#valid_file_ids = train_file_ids
class Corpus(object):
    def __init__(self, path):
        from pytorch_pretrained_bert import OpenAIGPTTokenizer
        tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
        self.train, self.train_sens, self.train_trees = self.tokenize(train_file_ids, tokenizer)
        self.valid, self.valid_sens, self.valid_trees = self.tokenize(valid_file_ids, tokenizer)
        self.test, self.test_sens, self.test_trees = self.tokenize(test_file_ids, tokenizer)
        self.rest, self.rest_sens, self.rest_trees = self.tokenize(rest_file_ids, tokenizer)

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
             words.append(w)
        return words

    def add_words(self, file_ids):
        # Add words to the dictionary
        for id in file_ids:
            sentences = ptb.parsed_sents(id)
            for sen_tree in sentences:
                words = self.filter_words(sen_tree)
                words = ['<s>'] + words + ['</s>']

    def tokenize(self, file_ids, gpt_tokenizer):

        def tree2list(tree):
            if isinstance(tree, nltk.Tree):
                if tree.label() in word_tags:
                    return tree.leaves()[0]
                else:
                    root = []
                    for child in tree:
                        c = tree2list(child)
                        if c != []:
                            root.append(c)
                    if len(root) > 1:
                        return root
                    elif len(root) == 1:
                        return root[0]
            return []

        sens_idx = []
        sens = []
        trees = []
        for id in file_ids:
            sentences = ptb.parsed_sents(id)
            for sen_tree in sentences:
                words = self.filter_words(sen_tree)
                words = ['<s>'] + words + ['<s>']
                # if len(words) > 50:
                #     continue
                ### now GPT tokenization
                words = gpt_tokenizer.tokenize(" ".join(words))
                ########
                sens.append(words)
                idx = gpt_tokenizer.convert_tokens_to_ids(words)
                sens_idx.append(torch.LongTensor(idx))
                trees.append(tree2list(sen_tree))

        return sens_idx, sens, trees
