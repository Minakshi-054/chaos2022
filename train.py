from silence_tensorflow import silence_tensorflow
silence_tensorflow()


import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
logging.getLogger('tensorflow').disabled = True

import numpy as np
from scipy.linalg import expm, sinm, cosm
from scipy import integrate
import random
import matplotlib.pyplot as plt
from skimage.util.shape import view_as_windows
from pathlib import Path
import tensorflow as tf
from sklearn.metrics import average_precision_score
import sys

from sklearn.metrics import classification_report

#CUDA_VISIBLE_DEVICES=3,4,5,6,7 python train.py 2 24 222 train

THRESHOLD = 0.5
tf.get_logger().setLevel('INFO')


limit = None
epoch = 20
num_gpu = 4
try:
    num_gpu = int(sys.argv[1])
    limit = None if int(sys.argv[2]) == -1 else int(sys.argv[2])#-1 means use all
    epoch = 2 if limit!=-1 else int(sys.argv[3])
    mode = sys.argv[4]#either train or test
except:
    raise Exception(
        f"Please Enter num_gpu 0 to 8 (default 8) as first arg, and -1 to train or any number to test the code, epoch number (20 default) and last input as mode 'train' or 'test'")

assert mode in ["train", "test"], "use last argument as train or test mode"
assert num_gpu < 9, "Number of GPU should be lesser than 9"

gpulist = [f"/gpu:{i}" for i in range(num_gpu)]

# strategy = tf.distribute.MultiWorkerMirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"])
strategy = tf.distribute.MirroredStrategy(devices=gpulist)

checkpoint_path = f"/data/stg60/CACHE/version_0/version_milton.ckpt"
root = Path(f"/data/stg60/DATA/version_0")
trainpath = root / 'train.npz'
valpath = root / 'val.npz'
testpath = root / 'par.npz'
predpath = root / 'pred.npz'

BATCH_SIZE = 128
TEST_BATCH_SIZE = BATCH_SIZE
SHUFFLE_BUFFER_SIZE = BATCH_SIZE * 128

print('Train Loading')
with np.load(trainpath, allow_pickle=True) as data:
    train_examples = data['x'][:limit]
    train_labels = (data['y'][:limit]).astype(np.int64)

train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE).prefetch(8)
print('Train Loaded')

print('Val Loading')
with np.load(valpath, allow_pickle=True) as data:
    val_examples = data['x']
    val_labels = (data['y']).astype(np.int64)

val_dataset = tf.data.Dataset.from_tensor_slices((val_examples, val_labels)).prefetch(8)
val_dataset = val_dataset.batch(BATCH_SIZE)
print('Val Loaded')


print('Test Loading')
with np.load(testpath, allow_pickle=True) as data:
    test_examples = data['x']
    test_labels = (data['y']).astype(np.int64)

test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels)).prefetch(8)
test_dataset = test_dataset.batch(TEST_BATCH_SIZE)
print('Test Loaded')

print('Pred Loading')
with np.load(predpath, allow_pickle=True) as data:
    pred_examples = data['x']
    pred_labels = (data['y']).astype(np.int64)

pred_dataset = tf.data.Dataset.from_tensor_slices((pred_examples, pred_labels)).prefetch(8)
pred_dataset = test_dataset.batch(TEST_BATCH_SIZE)
print('Pred Loaded')


options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
train_data = train_dataset.with_options(options)
val_dataset = val_dataset.with_options(options)
test_dataset = test_dataset.with_options(options)
pred_dataset = pred_dataset.with_options(options)





print("Creating a model")
tf.keras.backend.set_floatx('float64')



class TestResult(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        y_prob = tf.sigmoid(self.model.predict(test_examples, batch_size=TEST_BATCH_SIZE))
        y_out = (y_prob > 0.5).numpy().astype("int32")
        target_names = ['No-Fall', 'Fall']
        print("Results on Par Set")
        report = classification_report(test_labels, y_out, target_names=target_names, zero_division = 0)
        with open("reports/report_par_milton.txt","w") as f:
            f.write(report)
        print(report)
        
    def on_epoch_end(self, epoch, logs=None):
        y_prob = tf.sigmoid(self.model.predict(pred_examples, batch_size=TEST_BATCH_SIZE))
        y_out = (y_prob > 0.5).numpy().astype("int32")
        target_names = ['No-Fall', 'Fall']
        print("Results on Pred Set")
        report = classification_report(pred_labels, y_out, target_names=target_names, zero_division = 0)
        with open("reports/report_pred_milton.txt","w") as f:
            f.write(report)
        print(report)
        
        

with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(128, 4)),
        tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=0.1, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
        ),
        
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=False)),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(
                                            axis=-1,
                                            momentum=0.99,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            beta_initializer='zeros',
                                            gamma_initializer='ones',
                                            moving_mean_initializer='zeros',
                                            moving_variance_initializer='ones',
                                            beta_regularizer=None,
                                            gamma_regularizer=None,
                                            beta_constraint=None,
                                            gamma_constraint=None,
                                            ),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(
                                            axis=-1,
                                            momentum=0.99,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            beta_initializer='zeros',
                                            gamma_initializer='ones',
                                            moving_mean_initializer='zeros',
                                            moving_variance_initializer='ones',
                                            beta_regularizer=None,
                                            gamma_regularizer=None,
                                            beta_constraint=None,
                                            gamma_constraint=None,
                                            ),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.BatchNormalization(
                                            axis=-1,
                                            momentum=0.99,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            beta_initializer='zeros',
                                            gamma_initializer='ones',
                                            moving_mean_initializer='zeros',
                                            moving_variance_initializer='ones',
                                            beta_regularizer=None,
                                            gamma_regularizer=None,
                                            beta_constraint=None,
                                            gamma_constraint=None,
                                            ),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(
                                            axis=-1,
                                            momentum=0.99,
                                            epsilon=0.001,
                                            center=True,
                                            scale=True,
                                            beta_initializer='zeros',
                                            gamma_initializer='ones',
                                            moving_mean_initializer='zeros',
                                            moving_variance_initializer='ones',
                                            beta_regularizer=None,
                                            gamma_regularizer=None,
                                            beta_constraint=None,
                                            gamma_constraint=None,
                                            ),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics = ['accuracy'])
    
    if mode == "test":
        model.load_weights(checkpoint_path)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_best_only=True)

if mode == "train":
    print("model Fitting started")
    print("#"*200)
    print(f"TRAINING STATS : GPU-NUM : {num_gpu}, limit: {limit}, gpulist = {gpulist}, Epoch : {epoch}")
    print("#"*200)
    history = model.fit(train_dataset, 
                        validation_data=val_dataset, 
                        epochs=epoch, 
                        callbacks=[callback, 
                                cp_callback, 
                                TestResult(), 
                                tf.keras.callbacks.TensorBoard(
                                        log_dir=f'logs', histogram_freq=0, write_graph=True,
                                        write_images=False, update_freq='batch', profile_batch=2,
                                        embeddings_freq=0, embeddings_metadata=None
                                    )
                                    ]
                    )

    print("Model Fitting Done")
    print("Model testing Started")
    print(f"Test Performance Loss, Accuracy = {model.evaluate(test_dataset)}")
    print("Model Testing Done")
    
    try:    
        print("Model Evaluation Started Val Set")
        y_prob = tf.sigmoid(model.predict(val_examples))
        y_out = (y_prob > 0.5).numpy().astype("int32")
    except Exception as e:
        sys.exit(e)
        
    target_names = ['No-Fall', 'Fall']
    report = classification_report(val_labels, y_out, target_names=target_names, zero_division = 0)
    with open("reports/report_val_milton.txt","w") as f:
        f.write(report)
    print(report)
    
    
else:
    print("Model Evaluation Started")
    
    y_prob = tf.sigmoid(model.predict(test_examples))
    y_out = (y_prob > 0.84).numpy().astype("int32")

    target_names = ['No-Fall', 'Fall']
    report = classification_report(test_labels, y_out, target_names=target_names, zero_division = 0)
    
    with open("reports/test_report_lt10_0_85.txt","w") as f:
        f.write(report)
    
    print(report)



