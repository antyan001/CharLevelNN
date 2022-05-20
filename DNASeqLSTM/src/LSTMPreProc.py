
import os
import sys

curruser = os.environ.get('USER')
# sys.path.insert(0, './../src/')
# sys.path.insert(0, '/home/{}/notebooks/support_library/'.format(curruser))
sys.path.insert(0, '/home/{}/python36-libs/lib/python3.6/site-packages/'.format(curruser))

from pathlib import Path
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from Bio import SeqIO
from sklearn.preprocessing import LabelEncoder

import random
random.seed(42)

class preprocIO(object):

    def __init__(self):

        pass
        # self.SEQUENCE_LENGTH = SEQUENCE_LENGTH
        # self.step = step
        # self.right_len = right_len
        # self.left_len = left_len


    def streamFastaIO (self, path):
        patt = re.compile(r'^[^N][ATCG]+(?!N[ATCG]+)$')
        for sequence in SeqIO.parse(path,"fasta"):
            outSeqStr = str(sequence.seq)
            if len(patt.findall(outSeqStr))!=0:
                res = outSeqStr
            else:
                continue
            yield res

    def genSeqMoveRightChar(self, seqlst, SEQUENCE_LENGTH, step):
        next_chars = []
        sentences = []
        for seq in tqdm(seqlst):
            for i in range(0, len(seq) - SEQUENCE_LENGTH, step):
                leftpRightChunk = seq[i: i + SEQUENCE_LENGTH] + seq[i + SEQUENCE_LENGTH+1: i + int(2*SEQUENCE_LENGTH)+1]
                sentences.append(leftpRightChunk)
                next_chars.append(seq[i + SEQUENCE_LENGTH])

        return sentences, next_chars            
            

    def genSeqMoveRight(self, seqlst, SEQUENCE_LENGTH, step, char_indices):
        next_chars_indx = []
        sentences = []
        for seq in tqdm(seqlst):
            for i in range(0, len(seq) - SEQUENCE_LENGTH, step):
                leftpRightChunk = seq[i: i + SEQUENCE_LENGTH] + seq[i + SEQUENCE_LENGTH+1: i + int(2*SEQUENCE_LENGTH)+1]
                sentences.append(leftpRightChunk)
                next_chars_indx.append(char_indices[seq[i + SEQUENCE_LENGTH]])

        return sentences, next_chars_indx

    def RandomSamplingGen(self, seq, 
                                max_left_len=70, 
                                min_left_len = 10,
                                max_right_len=70, 
                                min_right_len = 10,                          
                                randsize_center = 30
                         ):
        
        indx_center = np.random.choice(np.arange(len(seq)), size=randsize_center)
#         indx_tail   = np.random.choice(np.arange(min_left_len,max_left_len+1), size=randsize_tail)
#         indx_head   = np.random.choice(np.arange(min_right_len,max_right_len+1), size=randsize_tail)
        
        chunkstuple=[]
        for i in indx_center:
            ri   = np.random.choice(np.arange(min_right_len,max_right_len+1), size=1)[0]
            li   = np.random.choice(np.arange(min_left_len,max_left_len+1), size=1)[0]
            if min_left_len<=len(seq[:i][-li:]):
                chunkstuple += [(seq[:i][-li:]+seq[i+1:i+ri+1],seq[i])]
        
        # chunkstuple = [(seq[:i],seq[i]) if (minlen<=len(seq[:i])<=left_len) else 
        #                (seq[-left_len:i],seq[i]) if (len(seq[:i])>left_len) else None for i in indx_center]
    
        return chunkstuple
    
    def getChunkesFromSeqToRight(self, seq, right_len=30, minlen = 5, lenc=None):
        i=minlen
        chunkstuple = []
        while True:
            if len(seq)-i >= right_len:
                left_chunk = seq[:i]
                if len(left_chunk) > right_len:
                    left_chunk = left_chunk[-right_len:]
                chunkstuple += [(' '.join([left_chunk, seq[i+1:i+right_len+1]]), lenc.transform([seq[i]])[0])]
                i+=1
            else:
                chunkstuple += [(' '.join([left_chunk, seq[i+1:len(seq)+1]]), lenc.transform([seq[i]])[0])]
                break
        return chunkstuple

    def getChunkesFromSeqToLeft(self, seq, left_len=30, minlen = 5, lenc = None):
        i = minlen
        chunkstuple = []
        while True:
            if len(seq)-i >= left_len:
                right_chunk = seq[-i:]
                if len(right_chunk) > left_len:
                    right_chunk = right_chunk[:left_len]
                chunkstuple += [(' '.join([seq[-(i+left_len+1):-(i+1):], right_chunk]), lenc.transform([seq[-(i+1)]])[0])]
                i+=1
            else:
                chunkstuple += [(' '.join([seq[-len(seq):-(i+1):], right_chunk]), lenc.transform([seq[-(i+1)]])[0])]
                break
        return chunkstuple

