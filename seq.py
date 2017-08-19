# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 12:41:25 2017

@author: maximilian_fritz
"""

import numpy as np
import pandas as pd
from static_property import static_property
import types
import scipy.signal as sps
import matplotlib.pylab as plt

class seq(str):
    """A sequence"""  
    
    @static_property
    def alphabet(self):
        return list(set(self))
        
    @static_property    
    def PSSM(self):
        return seq.outer(self.alphabet, self)

    @static_property
    def composition(self):
        return {char:self.count(char) for char in self.alphabet}

    @staticmethod
    def outer(seq1, seq2, chr_metric = lambda x, y: x==y):
        """
        Outer Product Function for sequences.
        USAGE CASES:
            seq.outer(s.alphabet, s) --> one-hot-encoding
            seq.outer(s, s, PAM.loc) --> construct álignment matrix
        """
        mat = np.zeros((len(seq1), len(seq2)))
        for i, chr1 in enumerate(seq1):
            for j, chr2 in enumerate(seq2):
                mat[i, j] = chr_metric(chr1, chr2)
        return pd.DataFrame(mat, index=seq1, columns=seq2)
         
    def map(self, fun_or_dic):
        """ 
        Map Function for sequences.
        USAGE CASES:
            s.map({c:c for c in s.alphabet})
            s.map(lambda c:c) 
            
        """
        
        if isinstance(fun_or_dic, types.FunctionType):
            return [fun_or_dic(char) for char in self] 
        elif isinstance(fun_or_dic, types.DictionaryType):
            return [fun_or_dic[char] for char in self]
            
    #map plotting(moving average), recurrence/fourier analysis
    def plot_map(self, fun_or_dic, window=1):
        array = np.array(self.map(fun_or_dic))
        conv = sps.convolve(array, 1.0/window**np.ones((window)))
        plt.plot(conv)
        plt.show()

class DNA(seq):
    """A DNA sequence"""

    _basecomplement = {'A': 'T', 'C': 'G', 'T': 'A', 'G': 'C'}  

    @static_property
    def complement(self):
        return ''.join(self.map(self._basecomplement))       
        
    @static_property
    def gc(self):
        return (self.composition['G']+self.composition['C']) * 100.0/len(self)        
     
 
class protein(seq):
    def __init__():
        pass
    #load aaindex
    
    
if __name__ == '__main__':    
    s = DNA('ACTGATCG')
    s.plot_map(lambda char: 1 if (char == 'A' or char == 'G') else 0, 2)
    
    
#regular expressions
#biopython, pygr, TAMO, pyHMM
#ORM, ZODB, PyTables