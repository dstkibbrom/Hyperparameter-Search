from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
import tensorflow as tf

data_set=pd.read_csv("ID_sequence.csv",index_col=False,skiprows=1, names=["arb_id"], sep=",")
data=data_set["arb_id"].values.tolist()
data=data[:100000]
# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = np.array(tokenizer.texts_to_sequences([data])[0],dtype=np.int32)-1 #The -1 is because tokenizer starts from 1
total_ids=len(set(encoded))



def create_dataset(encoded_data,seq_length, batch_size):
#     examples_per_epoch = len(encoded_data)//(seq_length+1)
    # Create training examples / targets
    ids_dataset = tf.data.Dataset.from_tensor_slices(encoded_data)
    sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

    def split_input_target(chunk):
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return input_ids, target_ids

    dataset = sequences.map(split_input_target)
    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    dataset = dataset.shuffle(3000).batch(batch_size, drop_remainder=True)
    return dataset


import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

HP_SEQ_LEN = hp.HParam('seq_len', hp.Discrete([15,20,25]))
HP_BATCH_SIZE = hp.HParam('batch_size', hp.Discrete([512,1024,2048]))
HP_RNN_TYPE = hp.HParam('RNN_type', hp.Discrete(['GRU', 'LSTM','peepholeLSTM']))
HP_RNN_NLAYERS = hp.HParam('RNN_nlayers', hp.Discrete([1, 2, 3]))
HP_RNN_UNITS = hp.HParam('RNN_units', hp.Discrete([64, 128, 256]))
HP_GRAD_CLIP= hp.HParam('grad_clip', hp.Discrete([1, 2,3]))

HP_LR=hp.HParam('lr', hp.Discrete([0.001,0.01,0.1]))
HP_MOMENTUM=hp.HParam('momentum', hp.Discrete([0.7,0.8,0.9]))
HP_EMBED_DIM = hp.HParam('EMBED_DIM', hp.Discrete([64, 128, 256]))
HP_STATE = hp.HParam('STATE', hp.Discrete([True,False])) 


METRIC_LOSS = 'val_loss'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
    hparams=[HP_SEQ_LEN,HP_BATCH_SIZE,HP_RNN_TYPE, 
             HP_RNN_NLAYERS,HP_RNN_UNITS, HP_GRAD_CLIP,
             HP_LR,HP_MOMENTUM,HP_EMBED_DIM, HP_STATE],
    metrics=[hp.Metric(METRIC_LOSS, display_name='Validation_Loss')],
)


def build_model(hparams):
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(total_ids, hparams[HP_EMBED_DIM], 
                                        batch_input_shape=[hparams[HP_BATCH_SIZE], None]))
    
    if hparams[HP_RNN_TYPE]=='GRU':
        for i in range(hparams[HP_RNN_NLAYERS]):
            model.add(tf.keras.layers.GRU(hparams[HP_RNN_UNITS], 
                                          kernel_initializer='glorot_uniform', 
                                          return_sequences=True,
                                          stateful=hparams[HP_STATE]))
        
    elif hparams[HP_RNN_TYPE]=='LSTM':
        for i in range(hparams[HP_RNN_NLAYERS]):
            model.add(tf.keras.layers.LSTM(hparams[HP_RNN_UNITS], 
                                          kernel_initializer='glorot_uniform', 
                                          return_sequences=True,
                                          stateful=hparams[HP_STATE]))
    else:
        for i in range(hparams[HP_RNN_NLAYERS]):
            model.add(tf.keras.layers.RNN(tf.keras.experimental.PeepholeLSTMCell(hparams[HP_RNN_UNITS], kernel_initializer='glorot_uniform'), return_sequences=True, stateful=hparams[HP_STATE]))
            
    model.add(tf.keras.layers.Dense(total_ids))
    
    learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(hparams[HP_LR], 1000, 0.1) # decay learnign rate in every 1000 steps
    opt=tf.keras.optimizers.SGD(learning_rate=learning_rate,momentum=hparams[HP_MOMENTUM],
                                clipvalue=hparams[HP_GRAD_CLIP])
    
    early_stopping_cb=tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    
    model.compile(optimizer=opt, loss=loss)
    train_ds=create_dataset(encoded[:int(len(encoded)*0.9)],hparams[HP_SEQ_LEN],hparams[HP_BATCH_SIZE])
    val_ds=create_dataset(encoded[int(len(encoded)*0.9):],hparams[HP_SEQ_LEN],hparams[HP_BATCH_SIZE])
    history = model.fit(train_ds, validation_data=val_ds, epochs=100,verbose=0, callbacks=[early_stopping_cb])
    return min(history.history['val_loss'][-5:])


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        loss_val = build_model(hparams)
        tf.summary.scalar(METRIC_LOSS, loss_val, step=1)

session_num = 0

for seq_len in HP_SEQ_LEN.domain.values:
    for batch_size in HP_BATCH_SIZE.domain.values:
        for rnn_type in HP_RNN_TYPE.domain.values:
            for rnn_nlayers in HP_RNN_NLAYERS.domain.values:
                for rnn_units in HP_RNN_UNITS.domain.values:
                    for grad_clip in HP_GRAD_CLIP.domain.values: 
                        for lr in HP_LR.domain.values: 
                            for momentum in HP_MOMENTUM.domain.values:  
                                for embedding_dim in HP_EMBED_DIM.domain.values: 
                                    for state in HP_STATE.domain.values: 
                                        hparams = {HP_SEQ_LEN: seq_len,
                                                   HP_BATCH_SIZE: batch_size,
                                                   HP_RNN_TYPE: rnn_type,
                                                   HP_RNN_NLAYERS: rnn_nlayers,
                                                   HP_RNN_UNITS: rnn_units,
                                                   HP_GRAD_CLIP:grad_clip,
                                                   HP_LR:lr,
                                                   HP_MOMENTUM:momentum,
                                                   HP_EMBED_DIM:embedding_dim,
                                                   HP_STATE:state
                                        }
                                        run_name = "run-%d" % session_num
                                        if session_num % 1000 ==0:
                                            print('--- Starting trial: %s' % run_name)
                                            print({h.name: hparams[h] for h in hparams})
                                        run('logs/hparam_tuning/' + run_name, hparams)
                                        session_num += 1

print(session_num) 

