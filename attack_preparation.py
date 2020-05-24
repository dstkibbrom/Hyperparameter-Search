import random
from tensorflow.keras.preprocessing import sequence
def create_dataset(encoded_data,seq_length, batch_size, anomaly_size):
    maxlen=seq_length+anomaly_size
    benign=tf.data.Dataset.from_tensor_slices(encoded_data[:int(0.34*len(encoded_data))]) #create dataset
    benign=benign.window(seq_length, shift=1, drop_remainder=True)  #slice the data into a windows of size seq_len+1
    benign_transformed=[]
    for benign_window in benign:
        def to_numpy(benign_ds):
            return list(benign_ds.as_numpy_iterator())
        benign_data=[to_numpy(benign_window)]            #paddsequence only works with double brackets
        benign_data=sequence.pad_sequences(benign_data,maxlen=maxlen).tolist()
        benign_transformed.append(benign_data[0])
        
    benign_dataset = tf.data.Dataset.from_tensor_slices(benign_transformed)
    def split_input_target(chunk):
        input_ids = chunk[:]
        target = [1,0,0]
        return input_ids, target
    benign_dataset = benign_dataset.map(split_input_target)
    
    
    insertion=tf.data.Dataset.from_tensor_slices(encoded_data[int(0.34*len(encoded_data)):2*int(0.34*len(encoded_data))]) #create dataset    
    insertion=insertion.window(seq_length, shift=1, drop_remainder=True)  #slice the data into a windows of size seq_len+1
    insertion_transformed=[]

    for insertion_window in insertion:
        def to_numpy(insertion_ds):
            return list(insertion_ds.as_numpy_iterator())
        insertion_data=to_numpy(insertion_window)
        for i in range(anomaly_size):
            insertion_data.insert(random.randrange(len(insertion_data)+1),random.randrange(max_id))
            
        insertion_data=sequence.pad_sequences([insertion_data],maxlen=maxlen).tolist()
        insertion_transformed.append(insertion_data[0])
    
#     insertion_dataset=tf.data.Dataset.from_tensor_slices(insertion_transformed)
#     insertion_dataset=insertion_dataset.window(1, shift=1, drop_remainder=True)
#     insertion_dataset=insertion_dataset.flat_map(lambda window:window.batch(seq_length)) #concerts it to a flat windows size sets
#     insertion_dataset = insertion_dataset.map(lambda windows: (windows[:, :], [0,1,0]))
    insertion_dataset = tf.data.Dataset.from_tensor_slices(insertion_transformed)
    def split_input_target(chunk):
        input_ids = chunk[:]
        target = [0,1,0]
        return input_ids, target
    insertion_dataset = insertion_dataset.map(split_input_target)
    
    drop=tf.data.Dataset.from_tensor_slices(encoded_data[int(-0.34*len(encoded_data)):]) #create dataset
    drop=drop.window(seq_length, shift=1, drop_remainder=True)  #slice the data into a windows of size seq_len+1
    drop_transformed=[]

    for drop_window in drop:
        def to_numpy(drop_ds):
            return list(drop_ds.as_numpy_iterator())
        drop_data=to_numpy(drop_window)
        for i in range(anomaly_size):
            drop_data.pop(random.randrange(len(drop_data)))
        drop_data=sequence.pad_sequences([drop_data],maxlen=seq_length+anomaly_size).tolist()
        drop_transformed.append(drop_data[0])
    
#     drop_dataset=tf.data.Dataset.from_tensor_slices(drop_transformed)
#     drop_dataset=drop_dataset.window(1, shift=1, drop_remainder=True)
#     drop_dataset=drop_dataset.flat_map(lambda window:window.batch(seq_length)) #concerts it to a flat windows size sets
#     drop_dataset = drop_dataset.map(lambda windows: (windows[:, :], [0,0,1]))
    
    drop_dataset = tf.data.Dataset.from_tensor_slices(drop_transformed)
    def split_input_target(chunk):
        input_ids = chunk[:]
        target = [0,0,1]
        return input_ids, target
    drop_dataset = drop_dataset.map(split_input_target)
    
    all_data=benign_dataset.concatenate(insertion_dataset)
    whole_data=drop_dataset.concatenate(all_data)
    whole_data=whole_data.shuffle(20000).batch(batch_size)
    
    
    return whole_data