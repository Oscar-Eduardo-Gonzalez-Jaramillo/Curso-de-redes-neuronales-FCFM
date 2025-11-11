import mlflow
import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.optimizers import Adam
import optuna
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from optuna.integration import TFKerasPruningCallback
import gc
from tensorflow.keras.applications import ResNet50, EfficientNetB0, MobileNetV2, DenseNet121
from tensorflow.keras.applications import InceptionV3, MobileNetV3Large
from functools import partial
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import shutil
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU(s) con memory growth activado.")
    except RuntimeError as e:
        print(e)

REPO_NAME = "Curso-de-redes-neuronales-FCFM"
REPO_OWNER = "Oscar-Eduardo-Gonzalez-Jaramillo"
USER_NAME = "Oscar-Eduardo-Gonzalez-Jaramillo"
os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = "bYkmar-ganxup-xowne7"
mlflow.set_tracking_uri(f'https://dagshub.com/{REPO_OWNER}/{REPO_NAME}.mlflow')

# Preprocesamiento adaptado
BASE_DIR = r"/tf/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(BASE_DIR, "images")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

df = pd.read_csv(TRAIN_CSV)
df['image_id'] = df['image_id'] + '.jpg'
# VERIFICAR DUPLICADOS
print(f"Total imágenes: {len(df)}")
print(f"Duplicados: {df['image_id'].duplicated().sum()}")
df = df.drop_duplicates(subset='image_id').reset_index(drop=True)
labels = ['healthy', 'multiple_diseases', 'rust', 'scab']
y = df[labels].values.astype(np.float32)

df_train, df_val, y_train, y_val = train_test_split(
    df, y,
    test_size=0.3,
    random_state=SEED,
    stratify=np.argmax(y, axis=1),
    shuffle=True
)
print(f"Train: {len(df_train)}, Val: {len(df_val)}")
print("Distribución en train:")
print(pd.Series(np.argmax(y_train, axis=1)).value_counts(normalize=True))
print("Distribución en val:")
print(pd.Series(np.argmax(y_val, axis=1)).value_counts(normalize=True))

# Class weights
class_counts = np.sum(y_train, axis=0)
total_samples = len(y_train)
num_classes = len(class_counts)
class_weights_dict = {}
for i, count in enumerate(class_counts):
    if count > 0:
        class_weights_dict[i] = total_samples / (num_classes * count)
    else:
        class_weights_dict[i] = 0.0
print("Class weights computed.")

def make_dataset(df, y_labels, is_training=True):
    file_paths = [os.path.join(IMG_DIR, img) for img in df['image_id']]
    labels = y_labels
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)  # [0, 255]
        if is_training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=32.0)  # Ajustado para [0,255]
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
        return img, label
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if is_training:
        dataset = dataset.shuffle(1024, seed=SEED, reshuffle_each_iteration=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = make_dataset(df_train, y_train, is_training=True)
val_ds = make_dataset(df_val, y_val, is_training=False)

print("Datasets prepared.")

def get_preprocess_func(arch):
    if arch == "EfficientNetB0":
        return tf.keras.applications.efficientnet.preprocess_input
    elif arch == "MobileNetV2":
        return tf.keras.applications.mobilenet_v2.preprocess_input
    elif arch == "ResNet50":
        return tf.keras.applications.resnet.preprocess_input
    elif arch == "DenseNet121":
        return tf.keras.applications.densenet.preprocess_input
    elif arch == "InceptionV3":
        return tf.keras.applications.inception_v3.preprocess_input
    elif arch == "MobileNetV3Large":
        return tf.keras.applications.mobilenet_v3.preprocess_input

def create_model(trial, num_classes, img_size=(224, 224)):
    arch = trial.suggest_categorical(
        "architecture",
        ["EfficientNetB0", "MobileNetV2", "ResNet50", "DenseNet121", "InceptionV3", "MobileNetV3Large"]
    )

    if arch == "EfficientNetB0":
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_size + (3,))
    elif arch == "MobileNetV2":
        base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=img_size + (3,))
    elif arch == "ResNet50":
        base_model = ResNet50(include_top=False, weights="imagenet", input_shape=img_size + (3,))
    elif arch == "DenseNet121":
        base_model = DenseNet121(include_top=False, weights="imagenet", input_shape=img_size + (3,))
    elif arch == "InceptionV3":
        base_model = InceptionV3(include_top=False, weights="imagenet", input_shape=img_size + (3,))
    elif arch == "MobileNetV3Large":
        base_model = MobileNetV3Large(include_top=False, weights="imagenet", input_shape=img_size + (3,))

    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    pooling_type = trial.suggest_categorical("pooling", ["avg", "max"])
    num_hidden_layers = trial.suggest_int("num_hidden_layers", 0, 1)
    freeze_ratio = trial.suggest_float("freeze_ratio", 0.80, 1.0)

    num_layers = len(base_model.layers)
    num_freeze = int(num_layers * freeze_ratio)
    for layer in base_model.layers[:num_freeze]:
        layer.trainable = False

    inputs = layers.Input(shape=img_size + (3,))
    x = inputs

    x = base_model(x)
    if pooling_type == "avg":
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalMaxPooling2D()(x)
    x = layers.Dropout(dropout)(x)

    for i in range(num_hidden_layers):
        n_units = trial.suggest_int(f"n_units_l{i}", 64, 128)
        x = layers.Dense(n_units, activation="relu")(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    tf.keras.backend.clear_session()
    gc.collect()
    return model

def objective(trial, train_dataset, val_dataset):
    with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
        run_id = mlflow.active_run().info.run_id
        trial.set_user_attr("run_id", run_id)

        arch = trial.suggest_categorical(
            "architecture",
            ["EfficientNetB0", "MobileNetV2", "ResNet50", "DenseNet121", "InceptionV3", "MobileNetV3Large"]
        )
        preprocess_func = get_preprocess_func(arch)


        local_train_ds = train_dataset.map(
            lambda img, label: (preprocess_func(img), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        local_val_ds = val_dataset.map(
            lambda img, label: (preprocess_func(img), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        model = create_model(trial, num_classes=4)
        mlflow.log_params(trial.params)

        earlystop = EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            restore_best_weights=True,
            patience=5,
            verbose=1
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_accuracy',
            mode='max',
            factor=0.5,
            patience=10,
            verbose=1
        )

        history = model.fit(
            local_train_ds,
            validation_data=local_val_ds,
            epochs=30,
            callbacks=[TFKerasPruningCallback(trial, "val_accuracy"), earlystop, lr_scheduler],
            verbose=1
        )

        val_acc = history.history['val_accuracy'][-1]

        del model
        tf.keras.backend.clear_session()
        gc.collect()

        return val_acc

mlflow.tensorflow.autolog(log_models=True)
mlflow.set_experiment("TAREA_6_ENFERMEDADES")

with mlflow.start_run(run_name="Architecture_Optimization_PlantPathology"):
    study = optuna.create_study(
        direction="maximize",
        study_name="Architecture_plantpathology",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        storage="sqlite:///Tarea_6_architecture_plantpath.db",
        load_if_exists=True
    )

    partial_objective = partial(objective, train_dataset=train_ds, val_dataset=val_ds)

    study.optimize(
        partial_objective,
        n_trials=5,
        callbacks=[]
    )

    best_trial = study.best_trial
    mlflow.log_params(best_trial.params)
    mlflow.log_metric("best_val_accuracy", best_trial.value)

    with mlflow.start_run(nested=True, run_name="Best_Model_Training"):

        best_arch = best_trial.params['architecture']
        preprocess_func = get_preprocess_func(best_arch)

        local_train_ds = train_ds.map(
            lambda img, label: (preprocess_func(img), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        local_val_ds = val_ds.map(
            lambda img, label: (preprocess_func(img), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        best_model = create_model(best_trial, num_classes=4)

        mlflow.log_params(best_trial.params)

        earlystop = EarlyStopping(
            monitor='val_accuracy',
            mode='max',
            restore_best_weights=True,
            patience=20,
            verbose=1
        )

        lr_scheduler = ReduceLROnPlateau(
            monitor='val_accuracy',
            mode='max',
            factor=0.5,
            patience=10,
            verbose=1
        )

        best_model.fit(
            local_train_ds,
            validation_data=local_val_ds,
            epochs=100,
            callbacks=[earlystop, lr_scheduler],
            verbose=1,
            class_weight=class_weights_dict
        )

        best_model.save("best_model.keras")
        mlflow.log_artifact("best_model.keras", artifact_path="best_model")
        os.remove("best_model.keras")

