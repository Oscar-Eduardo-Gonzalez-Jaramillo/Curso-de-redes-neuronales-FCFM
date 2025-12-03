import mlflow
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision
import zipfile
import glob

mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"Error en GPU config: {e}")

tf.keras.backend.clear_session()

REPO_NAME = "Curso-de-redes-neuronales-FCFM"
REPO_OWNER = "Oscar-Eduardo-Gonzalez-Jaramillo"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Oscar-Eduardo-Gonzalez-Jaramillo"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "850ff9069dde5c4a18ce975bcc338ba7ae73c758"
mlflow.set_tracking_uri(f"https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow")
mlflow.set_experiment("TAREA_7")

def parse_example(serialized_example):
    feature_description = {
        'image_jpeg': tf.io.FixedLenFeature([], tf.string),
        'attributes': tf.io.FixedLenFeature([40], tf.float32)
    }
    parsed = tf.io.parse_single_example(serialized_example, feature_description)
    img = tf.image.decode_jpeg(parsed['image_jpeg'], channels=3)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    attrs = parsed['attributes']
    return img, attrs

def augment_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    return img

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def create_raw_dataset(tfrecord_path, shuffle_buffer=2000, is_train=True):
    tfrecord_path = os.path.join(BASE_DIR, tfrecord_path)
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        dataset = dataset.shuffle(shuffle_buffer)
    return dataset

image_paths = glob.glob(os.path.join('/tf/oscar_images', '**/*.jpg'), recursive=True) + \
              glob.glob(os.path.join('/tf/oscar_images', '**/*.png'), recursive=True)

num_images = len(image_paths)
np.random.shuffle(image_paths)

train_size = int(0.8 * num_images)
val_size = int(0.1 * num_images)
test_size = num_images - train_size - val_size

train_paths = image_paths[:train_size]
val_paths = image_paths[train_size:train_size + val_size]
test_paths = image_paths[train_size + val_size:]

def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def load_augment(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [224, 224])
    img = tf.cast(img, tf.float32) / 255.0
    img = augment_image(img)
    return img, label


user_train_ds = tf.data.Dataset.from_tensor_slices((train_paths, [1.0] * len(train_paths)))
user_train_ds = user_train_ds.map(load_augment, num_parallel_calls=tf.data.AUTOTUNE)

user_val_ds = tf.data.Dataset.from_tensor_slices((val_paths, [1.0] * len(val_paths)))
user_val_ds = user_val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

user_test_ds = tf.data.Dataset.from_tensor_slices((test_paths, [1.0] * len(test_paths)))
user_test_ds = user_test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)

new_neg_dir = '/tf/Pruebas-familiares'
additional_paths = glob.glob(os.path.join(new_neg_dir, '**/*.jpg'), recursive=True) + \
                   glob.glob(os.path.join(new_neg_dir, '**/*.png'), recursive=True)

additional_neg_ds = tf.data.Dataset.from_tensor_slices((additional_paths, [0.0] * len(additional_paths)))
additional_neg_ds = additional_neg_ds.map(load_augment, num_parallel_calls=tf.data.AUTOTUNE)

raw_train = create_raw_dataset("tfrecords/celeba_train.tfrecord", is_train=True)
raw_val = create_raw_dataset("tfrecords/celeba_val.tfrecord", is_train=False)
raw_test = create_raw_dataset("tfrecords/celeba_test.tfrecord", is_train=False)

neg_train_celeba_ds = raw_train.map(lambda img, attrs: (augment_image(img), 0.0), num_parallel_calls=tf.data.AUTOTUNE).take(len(train_paths))
neg_train_ds = neg_train_celeba_ds.concatenate(additional_neg_ds)

neg_val_ds = raw_val.map(lambda img, attrs: (img, 0.0), num_parallel_calls=tf.data.AUTOTUNE).take(len(val_paths))
neg_test_ds = raw_test.map(lambda img, attrs: (img, 0.0), num_parallel_calls=tf.data.AUTOTUNE).take(len(test_paths))


train_ds = user_train_ds.concatenate(neg_train_ds).shuffle(2 * (len(train_paths) + len(additional_paths))).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = user_val_ds.concatenate(neg_val_ds).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = user_test_ds.concatenate(neg_test_ds).batch(32).prefetch(tf.data.AUTOTUNE)


num_pos = len(train_paths)
num_neg = len(train_paths) + len(additional_paths)
total = num_pos + num_neg
weight_for_0 = total / (2 * num_neg)
weight_for_1 = total / (2 * num_pos)
class_weights = {0: weight_for_0, 1: weight_for_1}


def weighted_bce(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    loss = bce(y_true, y_pred)
    weights = tf.gather([class_weights[i][1] for i in range(40)], tf.cast(y_true, tf.int32), axis=0)
    return tf.reduce_mean(loss * weights)


model = tf.keras.models.load_model("DenseNet_CelebA.h5", custom_objects={'weighted_bce': weighted_bce})

classifier_input = model.get_layer("classifier_logits").input
new_outputs = layers.Dense(1, activation="sigmoid", dtype='float32', name="oscar_classifier")(classifier_input)
new_model = tf.keras.Model(model.input, new_outputs)


new_model.compile(
    loss="binary_crossentropy",
    optimizer=Adam(0.0001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6),
]

mlflow.tensorflow.autolog()
with mlflow.start_run(run_name="DenseNet_Oscar_Face_With_Additional_Negs"):
    history = new_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
    )
    val_loss = min(history.history["val_loss"])
    mlflow.log_metric("min_val_loss", val_loss)
    test_loss, test_bin_acc, test_auc, test_prec, test_rec = new_model.evaluate(test_ds, verbose=1)
    mlflow.log_metric("test_binary_accuracy", test_bin_acc)
    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("test_precision", test_prec)
    mlflow.log_metric("test_recall", test_rec)
    mlflow.log_metric("test_loss", test_loss)
    new_model.save("DenseNet_Oscar.h5")
    mlflow.log_artifact("DenseNet_Oscar.h5")
