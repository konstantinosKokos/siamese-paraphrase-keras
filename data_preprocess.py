import numpy as np
import pickle
import spacy as sc
from tqdm import tqdm
import argparse

def create_datafiles(filename = 'ppdb-2.0-m-phrasal', max_sequence_length = 8, samples_per_file = 50000, nlp = sc.load('en_vectors_web_lg')):  
    '''
    given an input data file from PPDB, convert raw texts into vector sequences and store as a number of files
    Inputs: 
        - filename: the raw filename to process
        - max_sequence_length : the maximum words per sentence; smaller sentences will be zero-padded and larger sentences will be ignored
        - samples_per_file : the number of sentences per processed file (smaller -> more files, larger -> more RAM requirements)
    Stores:
        - a number of files containing tuple (X, Y), where X (Y) the vector sequences of the first (second) paraphrase pair elements
    '''
    vectorize = lambda x: np.stack(t.vector for i,t in enumerate(nlp(x)))
    expand = lambda x: np.pad(x, [[0, max_sequence_length - x.shape[0]], [0,0]], 
                          'constant')
    print('Starting.. Warning: Make sure you have enough space (>60GB) to store the full dataset!')
    xs = []
    ys = []
    with open(filename,'r') as f:
        lens = []
        linecount = 0
        filecount = 0
        for line in tqdm(f):
            s = line.split('|||')
            x = vectorize(s[1])
            y = vectorize(s[2])
            if x.shape[0] > max_sequence_length or y.shape[0] > max_sequence_length: continue
            x = expand(x)
            y = expand(y)
            linecount += 1
            if linecount > samples_per_file:
                # dump current array
                xs = np.array(xs)
                ys = np.array(ys)
                with open('dump_'+str(filecount), 'wb') as f:
                    print('Dumping {}...'.format(filecount))
                    print(xs.shape)
                    print(ys.shape)
                    pickle.dump((xs, ys), f)
                xs = []
                ys = []
                linecount = 1
                filecount += 1
            xs.append(x)
            ys.append(y)
        xs = np.array(xs)
        ys = np.array(ys)
        with open('dump_'+str(filecount), 'wb') as f:
            print('Dumping {}...'.format(filecount))
            print(xs.shape)
            print(ys.shape)
            pickle.dump((xs, ys), f)
    print('Done!')

def main():
    parser = argparse.ArgumentParser(description='Data preparation for siamese-paraphrase-keras')
    parser.add_argument('-f', type=str, help='name of raw data file', nargs='?')
    parser.add_argument('-m', type=int, help='max vector sequence length', nargs='?')
    parser.add_argument('-s', type=int, help='data samples per dump', nargs='?')
    args = parser.parse_args()
    f = '../data/ppdb-2.0-m-phrasal'
    s = 8
    m = 50000
    if args.f: f = args.f
    if args.m: m = args.m
    if args.s: s = args.s
    nlp = sc.load('en_vectors_web_lg')
    create_datafiles(f, m, s)
    
if __name__ == "__main__":
    main()