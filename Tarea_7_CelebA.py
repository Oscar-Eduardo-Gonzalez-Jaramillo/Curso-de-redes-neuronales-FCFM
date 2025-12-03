import mlflow
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras import mixed_precision


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

def create_dataset(tfrecord_path, batch_size=32, shuffle_buffer=2000, is_train=True):
    tfrecord_path = os.path.join(BASE_DIR, tfrecord_path)
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        dataset = dataset.shuffle(shuffle_buffer)
        dataset = dataset.map(lambda img, attrs: (augment_image(img), attrs), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = create_dataset("tfrecords/celeba_train.tfrecord", batch_size=32, is_train=True)
val_ds = create_dataset("tfrecords/celeba_val.tfrecord", batch_size=32, is_train=False)
test_ds = create_dataset("tfrecords/celeba_test.tfrecord", batch_size=32, is_train=False)

#train_ds = train_ds.take(500 // 32)
#val_ds = val_ds.take(100 // 32)
#test_ds = test_ds.take(100 // 32)

labels = []
for _, attrs in train_ds.take(-1):
    labels.append(attrs.numpy())
labels = np.concatenate(labels, axis=0)
class_weights = {}
for i in range(40):
    pos = np.sum(labels[:, i])
    neg = len(labels) - pos
    class_weights[i] = {0: pos / len(labels), 1: neg / len(labels)}


base_model = DenseNet121(include_top=False, weights=None, input_shape=(224,224,3))


inputs = layers.Input(shape=(224,224,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation="relu")(x)
outputs = layers.Dense(40, activation="sigmoid", dtype='float32', name="classifier_logits")(x)

model = tf.keras.Model(inputs, outputs)

def weighted_bce(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1)
    loss = bce(y_true, y_pred)
    weights = tf.gather([class_weights[i][1] for i in range(40)], tf.cast(y_true, tf.int32), axis=0)
    return tf.reduce_mean(loss * weights)

model.compile(
    loss=weighted_bce,
    optimizer=Adam(0.001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
        tf.keras.metrics.AUC(multi_label=True),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, min_lr=1e-6),

]


mlflow.tensorflow.autolog()
with mlflow.start_run(run_name="DenseNet121_CelebA"):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )

    val_loss = min(history.history["val_loss"])
    mlflow.log_metric("min_val_loss", val_loss)

    test_loss, test_bin_acc, test_auc, test_prec, test_rec = model.evaluate(test_ds, verbose=1)
    mlflow.log_metric("test_binary_accuracy", test_bin_acc)
    mlflow.log_metric("test_auc", test_auc)
    mlflow.log_metric("test_precision", test_prec)
    mlflow.log_metric("test_recall", test_rec)
    mlflow.log_metric("test_loss", test_loss)

    model.save("DenseNet-attr.keras")
    mlflow.log_artifact("DenseNet-attr.keras")

