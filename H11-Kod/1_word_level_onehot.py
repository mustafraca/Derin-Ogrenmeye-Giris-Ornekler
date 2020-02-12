# -*- coding: utf-8 -*-
"""
One-hot word embedding örneği
Programda verilen cümledeki farklı kelimeler bulunup,
her birine farklı bir indis atanıyor.
"""

import numpy as np

samples = ['The cat sat on the mat.', 
           'The dog ate my homework.']

token_index = {} #boş dictionary

print('Kelimelere atanan indisler:')
for sample in samples:
    for word in sample.split():# sıradaki kelimeyi al
        if word not in token_index:# token indeks verilmemişse            
            token_index[word] = len(token_index) + 1 # sıraki tamsayıyı ata
            print('\t',word,'=',token_index[word])
            
max_length = 10
results = np.zeros(shape=(len(samples),
                          max_length,
                          max(token_index.values()) + 1))

print('Kelimelere vektör ata:')
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        print(word,'=',token_index[word])
        results[i, j, index] = 1.
        print('\t',results[i,j,:])
           

#print(i,j,sample,word)