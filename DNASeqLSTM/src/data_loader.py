import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import sentencepiece as spm
# from bpe import Encoder
import pandas as pd
import numpy as np
import itertools
import collections
from tqdm import tqdm
from . import utils


class MyDataset(Dataset):
    def __init__(self, args):
        self.data_path = args.data_path
        self.max_rows  = args.max_rows
        self.chunksize = args.chunksize
        self.encoding  = args.encoding
        self.sep       = args.sep        
        self.usembedding       = args.usembedding
        self._embedding        = args._embedding
        self.embedlength       = args.embedlength
        self.embedAfterBatches = args.embedAfterBatches
        self.useBOCNGrams      = args.useBOCNGrams
        self.useSentencePieceTokenizer  = args.useSentencePieceTokenizer
        self.useNGramBPETokenizer       = args.useNGramBPETokenizer
        self.data_path_to_SentpBPE  = args.data_path_to_SentpBPE
        self.data_path_to_NGramBPE  = args.data_path_to_NGramBPE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       
        if args.extra_characters != []:
            self.vocabulary = list(args.alphabet) + list(args.extra_characters)
            self.number_of_characters = args.number_of_characters + \
                len(args.extra_characters)
        else:
            self.vocabulary = list(args.alphabet)
            self.number_of_characters = args.number_of_characters

        self.max_length = args.max_length
        self.num_classes = args.number_of_classes

        self.preprocessing_steps = args.steps

        self.identity_mat = np.identity(self.number_of_characters)
        texts, labels = [], []

        # chunk your dataframes in small portions
        chunks = pd.read_csv(self.data_path,
                             usecols=[args.text_column, args.label_column],
                             chunksize=self.chunksize,
                             sep=self.sep,
                             encoding=self.encoding,
                             nrows=self.max_rows,
                             na_filter=False)
        for df_chunk in tqdm(chunks, total=self.max_rows//self.chunksize+1):
            df_chunk.fillna('',inplace=True)
            df_chunk = df_chunk[df_chunk[args.text_column]!='']
            if self.preprocessing_steps is not None:
                df_chunk['processed_text'] = (df_chunk[args.text_column]
                                              .map(lambda text: utils.process_text(self.preprocessing_steps, text)))
                texts += df_chunk['processed_text'].tolist()
                labels += df_chunk[args.label_column].tolist()
            else:
                texts += df_chunk[args.text_column].tolist()
                labels += df_chunk[args.label_column].tolist()

        print('data loaded successfully with {0} rows'.format(len(labels)))

        self.texts = texts
        self.labels = labels
        self.length = len(self.labels)
        
    @staticmethod    
    def _gen_ngrams():
        """
          Generates all trigrams for characters from `trigram_chars`
          Using BOW relults in language dictionary scaling and it becomes a problem 
          as the current vocabulary size is large and addition of new words can make the problem worse.
          Hashing is an effective way to alleviate the open vocabulary problems in neural machine translation.
          We break down each word to a bag of char-trigrams. As the number of char-trigrams are fixed and small, 
          using a bag-of-CharTriGrams can be a good solution to this problem. 
          To get the char-trigrams for a word, we first append ‘#’ to both ends of the word 
          and then spilt it into tri-grams.            
        """
        trigram_chars= list("ACGT")#self.vocabulary #"ACGT"
        t4=[''.join(x) for x in itertools.product(trigram_chars,repeat=4)] #len(words)>=4        
        t3=[''.join(x) for x in itertools.product(trigram_chars,repeat=3)] #len(words)>=3
        t2=[''.join(x) for x in itertools.product(trigram_chars,repeat=2)] #len(words)>=2
        t1=[''.join(x) for x in itertools.product(trigram_chars,repeat=1)] #len(words)>=1
#         t2_start=['#'+''.join(x) for x in itertools.product(trigram_chars,repeat=2)] #len(words)==2
#         t2_end=[''.join(x)+'#' for x in itertools.product(trigram_chars,repeat=2)] #len(words)==2
#         t1=['#'+''.join(x)+'#' for x in itertools.product(trigram_chars)] #len(words)==1
        trigrams=t4+t3+t2+t1
        vocab_size=len(trigrams)
        trigram_map=dict(zip(trigrams,range(1,vocab_size+1))) # trigram to index mapping, indices starting from 1
        return trigram_map, vocab_size       
       
    def __len__(self):
        return self.length

    def __getitem__(self, index):      
        raw_text = self.texts[index]
        
        if self.usembedding and not self.embedAfterBatches:
            
            if self.useBOCNGrams:
                #################################################################
                ### Apply Word Hashing using a bag-of-Char-Uni/Bi/Tri/TertraGrams
                #################################################################
                trigram_map, _ = MyDataset._gen_ngrams()
                trigram_BOW = np.zeros((self.max_length,self.embedlength),dtype=np.float32)
                word=raw_text
                step = 0
#                 indices=collections.defaultdict(int)
                for shift in range(1,5):
                    for k in range(len(word)-(shift-1)): # generate all uni/bi/trigrams for word `word` and update `indices`
                        trig=word[k:k+shift]
                        idx=trigram_map.get(trig, 0)
                        if idx!=0:
                            if isinstance(self._embedding, nn.modules.sparse.Embedding):  
                                indx = torch.LongTensor([idx]).to(self.device)
                                ten = self._embedding(indx)
                                numpyEmbed = ten.cpu().detach().numpy().squeeze(0)
                            elif isinstance(self._embedding, np.ndarray):
                                numpyEmbed = self._embedding[idx]
                            trigram_BOW[step,:]=numpyEmbed
                        step+=1
                data = trigram_BOW

                if len(data) > self.max_length:
                    data = data[:self.max_length]
                elif 0 < len(data) < self.max_length:
                    data = np.concatenate(
                        (data, np.zeros((self.max_length - len(data), self.embedlength), dtype=np.float32)))
                elif len(data) == 0:
                    data = np.zeros(
                        (self.max_length, self.embedlength), dtype=np.float32)        
            elif self.useSentencePieceTokenizer:
                ####################################################################################
                ### Apply SentencePiece Python Wrapper with pretrained BPE model
                ### Use One-Hot as main metadata and concat BPE tokenization as additional meta-tail
                ####################################################################################
                sp = spm.SentencePieceProcessor()
                sp.Load(self.data_path_to_SentpBPE)            
                vocabSize = sp.GetPieceSize()
                data=np.zeros((self.max_length, self.embedlength),dtype=np.float32)
#                 pos=0
#                 for i, char in enumerate(raw_text):
#                     if char in self.vocabulary:
#                         idx = self.vocabulary.index(char)
#                         if isinstance(self._embedding, nn.modules.sparse.Embedding): 
#                             indx = torch.LongTensor([idx]).to(self.device)
#                             ten = self._embedding(indx)
#                             numpyEmbed = ten.cpu().detach().numpy().squeeze(0) 
#                         elif isinstance(self._embedding, np.ndarray):
#                             numpyEmbed = self._embedding[idx]                            
#                         data[pos,:] = numpyEmbed
#                         pos+=1    
                encIds = sp.EncodeAsIds(raw_text)
                #############################################
                ### Concat idexes from BPE tokenization 
                #############################################
#                 shift = len(self.vocabulary)-1
                for pos, val in enumerate(encIds):
                    idx = val
                    if idx!=499:
                        if isinstance(self._embedding, nn.modules.sparse.Embedding):
                            indx = torch.LongTensor([idx]).to(self.device)
                            ten = self._embedding(indx)
                            numpyEmbed = ten.cpu().detach().numpy().squeeze(0)
                        elif isinstance(self._embedding, np.ndarray):
                            numpyEmbed = self._embedding[idx]                        
                        data[pos,:] = numpyEmbed
#                     pos+=1
                if len(data) == 0:
                    data = np.zeros((self.max_length, self.embedlength), dtype=np.float32)
            elif self.useNGramBPETokenizer:
                #############################################################
                ### BPE Decoder: Use BPE vocab and tokenize raw data based on 
                ### specified ngrams range
                #############################################################
                bpe = Encoder.load(self.data_path_to_NGramBPE)
                vocabSize = bpe.vocab_size
                data=np.zeros((self.max_length, self.embedlength),dtype=np.float32)
                bpe_dct = bpe.learn_bpe_vocab([raw_text])
                bpe_dct.pop('__sow')
                bpe_dct.pop('__eow')
                encIds = [v for k,v in bpe_dct.items()]                
                for pos, val in enumerate(encIds):
                    idx = val
                    if isinstance(self._embedding, nn.modules.sparse.Embedding):
                        indx = torch.LongTensor([idx]).to(self.device)
                        ten = self._embedding(indx)
                        numpyEmbed = ten.cpu().detach().numpy().squeeze(0)
                    elif isinstance(self._embedding, np.ndarray):
                        numpyEmbed = self._embedding[idx]                        
                    data[pos,:] = numpyEmbed
#                     pos+=1
                if len(data) == 0:
                    data = np.zeros((self.max_length, self.embedlength), dtype=np.float32)                
            else:
                ############################################################
                ### Use basic Vocab of Characters and Apply One-Hot-Encoding
                ############################################################
#                 data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
#                                 dtype=np.float32)
                data=np.zeros((self.max_length, self.embedlength),dtype=np.float32)
                pos=0
                for i, char in enumerate(raw_text):
                    if char in self.vocabulary:
                        idx = self.vocabulary.index(char)
                        if isinstance(self._embedding, nn.modules.sparse.Embedding):
                            indx = torch.LongTensor([idx]).to(self.device)
                            ten = self._embedding(indx)
                            numpyEmbed = ten.cpu().detach().numpy().squeeze(0)
                        elif isinstance(self._embedding, np.ndarray):
                            numpyEmbed = self._embedding[idx]                            
                        data[pos,:] = numpyEmbed
                        pos+=1 
                if len(data) > self.max_length:
                    data = data[:self.max_length]
                elif 0 < len(data) < self.max_length:
                    data = np.concatenate(
                        (data, np.zeros((self.max_length - len(data), self.embedlength), dtype=np.float32)))
                elif len(data) == 0:
                    data = np.zeros(
                        (self.max_length, self.embedlength), dtype=np.float32)

        else:
        
            if self.useBOCNGrams:
                #################################################################
                ### Apply Word Hashing using a bag-of-Char-Uni/Bi/Tri/TertraGrams
                #################################################################
                trigram_map, vocab_size = MyDataset._gen_ngrams()
                trigram_BOW=np.zeros((self.max_length,vocab_size+1),dtype=np.float32)
                word=raw_text
                step = 0
                indices=collections.defaultdict(int)
                for shift in range(1,5):
                    for k in range(len(word)-(shift-1)): # generate all uni/bi/trigrams for word `word` and update `indices`
                        trig=word[k:k+shift]
                        idx=trigram_map.get(trig, 0)
                        #print(trig,idx)
                        if idx!=0:
                            ### CountVectorizer step vs One-Hot
                            indices[idx]+=1
                            trigram_BOW[step,idx]=indices[idx]
                        step+=1
                data = trigram_BOW

                if len(data) > self.max_length:
                    data = data[:self.max_length]
                elif 0 < len(data) < self.max_length:
                    data = np.concatenate(
                        (data, np.zeros((self.max_length - len(data), vocab_size+1), dtype=np.float32)))
                elif len(data) == 0:
                    data = np.zeros(
                        (self.max_length, vocab_size+1), dtype=np.float32)        
            elif self.useSentencePieceTokenizer:
                ###########################################################################################
                ### Apply SentencePiece Python Wrapper with pretrained BPE model
                ### Transform tokens' ids encoded with BPE model into Count-Hot Matrix: aka CountVectorizer  
                ###########################################################################################
                sp = spm.SentencePieceProcessor()
                sp.Load(self.data_path_to_SentpBPE)            
                vocabSize = sp.GetPieceSize()
                data=np.zeros((self.max_length, vocabSize),dtype=np.float32)
#                 pos=0
#                 for i, char in enumerate(raw_text):
#                     if char in self.vocabulary:
#                         data[pos,self.vocabulary.index(char)] = 1
#                         pos+=1   
                #############################################
                ### Concat idexes from BPE tokenization 
                #############################################
#                 shift = len(self.vocabulary)-1
                indices=collections.defaultdict(int)
                encIds = sp.EncodeAsIds(raw_text)
                for pos, idx in enumerate(encIds):
                    if idx!=499:
                        ### CountVectorizer step vs One-Hot
                        indices[idx]+=1
                        data[pos,idx] = indices[idx]
#                     pos+=1
                if len(data) == 0:
                    data = np.zeros((self.max_length, vocabSize), dtype=np.float32) 
            elif self.useNGramBPETokenizer:
                #############################################################
                ### BPE Decoder: Use BPE vocab and tokenize raw data based on 
                ### the specified ngrams range
                #############################################################
                bpe = Encoder.load(self.data_path_to_NGramBPE)
                vocabSize = bpe.vocab_size          
                data=np.zeros((self.max_length, vocabSize),dtype=np.float32)
                bpe_dct = bpe.learn_bpe_vocab([raw_text])
                bpe_dct.pop('__sow')
                bpe_dct.pop('__eow')
                encIds = [v for k,v in bpe_dct.items()]    
                indices=collections.defaultdict(int)
                for pos, idx in enumerate(encIds):
                    ### CountVectorizer step vs One-Hot
                    indices[idx]+=1
                    data[pos,idx] = indices[idx]
                if len(data) == 0:
                    data = np.zeros((self.max_length, vocabSize), dtype=np.float32)                    
            else: 
                ############################################################
                ### Use basic Vocab of Characters and Apply One-Hot-Encoding
                ############################################################
                data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                                dtype=np.float32)
                if len(data) > self.max_length:
                    data = data[:self.max_length]
                elif 0 < len(data) < self.max_length:
                    data = np.concatenate(
                        (data, np.zeros((self.max_length - len(data), self.number_of_characters), dtype=np.float32)))
                elif len(data) == 0:
                    data = np.zeros(
                        (self.max_length, self.number_of_characters), dtype=np.float32)

        label = self.labels[index]
        return data, label
