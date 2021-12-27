# import library
import math, os
from tqdm import tqdm
from shutil import copyfile
from data_process import *
from model import *

# 使用 TPU，檢測TPU是否有被使用
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() 

REPLICAS = strategy.num_replicas_in_sync

print("REPLICAS: ", REPLICAS)


# Data ##########################################
## Define Save and Load Data Location #################
if REPLICAS > 1:
    # 使用TPU訓練的話，數據必須存放在 Google Cloud Storage bucket
    from kaggle_datasets import KaggleDatasets

    GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')
    print(GCS_DS_PATH) # what do gcs paths look like?

    BATCH_SIZE_PER_REPLICA = 64
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    
else:
    GCS_DS_PATH = './Data'
    GLOBAL_BATCH_SIZE = 64
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

GCS_PATH = GCS_DS_PATH + '/tfrecords-jpeg-224x224' #192, 224, 331, 512 with IMAGE_SIZE in data_process
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec')

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
print("="*100)
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
print("="*100)

## Load Data ###################################
ds_train = get_training_dataset(TRAINING_FILENAMES, GLOBAL_BATCH_SIZE)
ds_valid = get_validation_dataset(VALIDATION_FILENAMES, GLOBAL_BATCH_SIZE)
ds_test = get_test_dataset(TEST_FILENAMES, GLOBAL_BATCH_SIZE)

print("="*100)
print("Training:", ds_train)
print ("Validation:", ds_valid)
print("Test:", ds_test)
print("="*100)
###############################################

EPOCHS = 1

# Model 的 loss、優化器定義 ##########################
with strategy.scope():
    model = build_model(class_num=len(CLASSES), image_size = IMAGE_SIZE) # CLASSES was defined in class_define
    model.summary()
    
  # Set reduction to `none` so we can do the reduction afterwards and divide by
  # global batch size.
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction=tf.keras.losses.Reduction.NONE
    )
    def compute_loss(labels, predictions):
        per_example_loss = loss_object(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)
    

    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.1,
        decay_steps=100,
        decay_rate=0.3,
        staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
###############################################

# 開始訓練 #######################################
def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss 

@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

dist_train_dataset = strategy.experimental_distribute_dataset(ds_train)
for epoch in range(EPOCHS):
    # TRAIN LOOP
    total_loss = 0.0
    num_batches = 0
    for x in tqdm(dist_train_dataset):
        total_loss += distributed_train_step(x)
        num_batches += 1
    train_loss = total_loss / num_batches
  
    template = ("Epoch {}, Loss: {}, Accuracy: {}")
    print (template.format(epoch+1, train_loss, train_accuracy.result()*100), flush=True)
  
    train_accuracy.reset_states()
###############################################

# 進行預測 ######################################
def test_step(inputs):
    images, idnum = inputs
    
    output = model(images, training=False)
    pred = tf.argmax(output, axis=1)
    return idnum, pred

@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))

data_list = []
for x in tqdm(ds_test):
    idnum, pred = distributed_test_step(x)

    if REPLICAS > 1:
        idnum_np, pred_np = idnum.values[0].numpy(), pred.values[0].numpy() # multi TPU
    else:
        idnum_np, pred_np = idnum.numpy(), pred.numpy()
        
    for i in range(idnum_np.shape[0]):
        data_list.append({"id":idnum_np[i].decode(), "label":pred_np[i]})
###############################################

        
# 提交答案 ######################################
from generate_csv import *
if REPLICAS > 1:
    generate_result(data_list, True)
else:
    generate_result(data_list, False)
###############################################