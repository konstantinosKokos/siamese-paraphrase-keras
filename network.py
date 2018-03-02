from keras.layers import Dense, LSTM, Lambda, Input, Concatenate, Flatten, Reshape
from keras.models import Model
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt

import pickle

# ==============================================================================================================
# DATA GENERATOR
# ==============================================================================================================

one_shuffler = lambda x: np.array([x[i][:] for i in np.random.permutation(np.arange(x.shape[0]))]) # shuffle the words around

def one_swapper(x, xs):
    """
    given an input vector sequence x and a matrix of vector sequences xs, randomly swap one vector element of x
    """
    newx = np.array(x, copy=True)
    newx[np.random.randint(x.shape[0]),:] = xs[np.random.randint(xs.shape[0]), np.random.randint(xs.shape[1]), :]
    return newx

def read_data(dissimilarity_ratio, batch_size):
    """
    read_data : a generator that iterates over files and performs the shuffling / swapping in real-time to produce the network's training data
    Inputs:
        - dissimilarity_ratio : [0,1) the percentage of paraphrase to non-paraphrase pairs
        - batch_size : the number of samples to yield per call
    """
    current_sample = 0
    current_file = 0
    x0, x0sh, x0sw, x1, x1sh, x1sw, y = [], [], [], [], [], [], []

    with open('..data/dump_'+str(current_file), 'rb') as f:
        x0s, x1s = pickle.load(f)        
    while True:
        if current_sample > x0s.shape[0]:
            current_sample = 0
            current_file += 1
            try:
                with open('data/dump_'+str(current_file), 'rb') as f:
                    x0s, x1s = pickle.load(f)
            except:
                current_file = 0
                with open('data/dump_'+str(current_file), 'rb') as f:
                    x0s, x1s = pickle.load(f)
                    
        next_x0 = x0s[current_sample, :, :] # original 0
        next_x0sh = one_shuffler(next_x0)
        next_x0sw = one_swapper(next_x0, x0s)
        
        if np.random.random() > dissimilarity_ratio: 
            pair_index = current_sample
            next_y = 1
        else: 
            pair_index = np.random.randint(x1s.shape[0])
            next_y = 0
        next_x1 = x1s[pair_index, :, :] # original 1
        next_x1sh = one_shuffler(next_x1)
        next_x1sw = one_swapper(next_x1, x1s)
        
        if len(x0) == batch_size: 
            yield  [np.array(x0), np.array(x0sh), np.array(x0sw), 
                    np.array(x1), np.array(x1sh), np.array(x1sw)], np.array(y)
            x0, x0sh, x0sw, x1, x1sh, x1sw, y = [], [], [], [], [], [], []
        else:
            x0.append(next_x0)
            x0sh.append(next_x0sh)
            x0sw.append(next_x0sw)
            x1.append(next_x1)
            x1sh.append(next_x1sh)
            x1sw.append(next_x1sw)
            y.append(next_y)
            
# ==============================================================================================================
# UTILITY WRAPPERS 
# ==============================================================================================================

# we define semantic and syntactic wrappers individually for ease of monitoring
def semantic_wrapper(po, psw):
    def semantic_metric(y_true, y_pred):
        return K.mean(K.maximum(0., 1 - po + psw))
    return semantic_metric
def syntactic_wrapper(po, psh):
    def syntactic_metric(y_true, y_pred):
        return K.mean(K.maximum(0., 1- po + psh))
    return syntactic_metric
def full_wrapper(po, psw, psh):
    def full_loss(y_true, y_pred):
        return K.mean(K.maximum(0., 1 - po + psh)) + K.mean(K.maximum(0., 1 - po + psw))
    return full_loss

# ==============================================================================================================
# NETWORK
# ==============================================================================================================

def build_siamese(max_sequence_length = 8, penalty_factor = 1, opt='adam', loss='mse'):
    """
    build_siamese : constructs the siamese paraphrase detection architecture
    INPUTS:
        - max_sequence_length : the maximum vectors per sequence (make sure it matches your training data)
        - penalty_factor : the weight assigned to the plausibility loss
    """
    
    # INPUTS
    original_0 = Input(shape=(max_sequence_length,300,)) # the first paraphrase pair element
    shuffled_0 = Input(shape=(max_sequence_length,300,)) # the first paraphrase pair element shuffled
    swapped_0 = Input(shape=(max_sequence_length,300,)) # the first paraphrase pair element swapped
    original_1 = Input(shape=(max_sequence_length,300,)) # the second paraphrase pair element
    shuffled_1 = Input(shape=(max_sequence_length,300,)) # the second paraphrase pair element shuffled
    swapped_1 = Input(shape=(max_sequence_length,300,)) # the second paraphrase pair element swapped

    # RECURRENT FUNCTIONS (these can be an arbitrary recurrent network)
    recurrent_function_0 = LSTM(301, name='recurrent_0')
    recurrent_function_1 = LSTM(301, name='recurrent_1')

    # SPLITTER
    # Keras doesn't allow tensor splitting, so we wrap the outputs with two Lambda layers
    composition =  Lambda(lambda x: x[:,:300], name='composition')
    plausibility = Lambda(lambda x: x[:,300:301], name='plausibility')

    # OUTPUTS
    ro_0 = recurrent_function_0(original_0) # the first twin's output to the first input
    rsh_0 = recurrent_function_0(shuffled_0) # the first twin's output to the shuffled first input
    rsw_0 = recurrent_function_0(swapped_0) # the first twin's output to the swapped first input
    co_0 = composition(ro_0) # the sentence representation of the first input
    po_0 = plausibility(ro_0) # the plausibility of the first input
    psh_0 = plausibility(rsh_0) # the plausibility of the shuffled first input
    psw_0 = plausibility(rsw_0) # the plausibility of the swapped first input

    ro_1 = recurrent_function_1(original_1) # similarly for the second twin / input ...
    rsh_1 = recurrent_function_1(shuffled_1)
    rsw_1 = recurrent_function_1(swapped_1)
    co_1 = composition(ro_1)
    po_1 = plausibility(ro_1)
    psh_1 = plausibility(rsh_1)
    psw_1 = plausibility(rsw_1)

    # BINDING
    merged = Concatenate()([co_0, co_1]) 
    cos = Lambda(lambda x: K.sum( K.l2_normalize(x[:,:300], axis=-1) * K.l2_normalize(x[:,300:600], axis=-1), axis=-1), 
                 (1,), name='cosine') (merged) # the cosine distance between the two networks' sentence representations
    cos = Reshape((1,))(cos) 
    e_f = Dense(1, activation='sigmoid')(cos) # wrapped in a simple regression layer to predict paraphrasing   

    # INTERMEDIATE LOSSES
    intermediate_loss_0 = K.mean(K.maximum(0., 1 - po_0 + psh_0)) + K.mean(K.maximum(0., 1 - po_0 + psw_0))
    intermediate_loss_1 = K.mean(K.maximum(0., 1 - po_1 + psh_1)) + K.mean(K.maximum(0., 1 - po_1 + psw_1))
    additive_loss = (intermediate_loss_0 + intermediate_loss_1) * penalty_factor

    # CONSTRUCTION
    siamese = Model(inputs=[original_0, shuffled_0, swapped_0, 
                            original_1, shuffled_1, swapped_1],
                    outputs = [e_f])
    
    siamese.add_loss(additive_loss) # we add the intermediate loss post-construction
    
    siamese.compile(optimizer=opt, loss = loss, 
                    metrics=['mse', # since the hinge losses are wrapped in different functions, we can monitor them during training 
                             semantic_wrapper(po_0, psw_0),
                             semantic_wrapper(po_1, psw_1),
                             syntactic_wrapper(po_0, psh_0),
                             syntactic_wrapper(po_1, psh_1)])
    
    siamese.summary()
    
    return siamese

def train_siamese(siamese, dis = 0.5, bs = 50, spe = 1000, ne = 100):
    # we call our generator to handle data preprocessing and network feeding
    h = siamese.fit_generator(read_data(dis, batch_size=bs), steps_per_epoch=spe, nb_epoch=ne)
    return h

def histplot(history):
    plt.figure(figsize=(13,5))
    plt.plot((np.array(history['semantic_metric_1'])), color='green')
    plt.plot((np.array(history['syntactic_metric_1'])), color='blue')
    plt.legend(['Semantic Loss', 'Syntactic Loss'])
    plt.title('Twin 1 Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.figure(figsize=(13,5))
    plt.plot((np.array(history['semantic_metric_2'])), color='green')
    plt.plot((np.array(history['syntactic_metric_2'])), color='blue')
    plt.legend(['Semantic Loss', 'Syntactic Loss'])
    plt.title('Twin 2 Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.figure(figsize=(13,5))
    plt.plot((np.array(history['loss'])), color='red')
    plt.plot((np.array(history['mean_squared_error'])), color='black')
    plt.legend(['Cumulative Loss', 'Prediction Loss'])
    plt.title('Siamese Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()