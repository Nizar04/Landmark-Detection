# config.py
from pathlib import Path

class Config:
    RANDOM_SEED = 42
    BASE_PATH = Path("data")
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    VALIDATION_SPLIT = 0.2
    NUM_CLASSES = 1020
    
    MODEL_PARAMS = {
        "dropout_rate": 0.5,
        "use_batch_norm": True,
        "backbone": "efficientnet_v2_b0",
        "freeze_backbone": False
    }
    
    DATA_AUGMENTATION = {
        "rotation_range": 20,
        "width_shift_range": 0.2,
        "height_shift_range": 0.2,
        "horizontal_flip": True,
        "fill_mode": "nearest"
    }

# data_loader.py
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config

class LandmarkDataLoader:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.label_encoder = None
        
    def prepare_data(self):
        self.encode_labels()
        train_df, val_df = train_test_split(
            self.df,
            test_size=Config.VALIDATION_SPLIT,
            stratify=self.df["landmark_id"],
            random_state=Config.RANDOM_SEED
        )
        return train_df, val_df
    
    def encode_labels(self):
        self.label_encoder = tf.keras.layers.StringLookup(
            num_oov_indices=0,
            vocabulary=self.df["landmark_id"].unique()
        )
    
    def create_dataset(self, dataframe, training=False):
        ds = tf.data.Dataset.from_tensor_slices(
            (dataframe["id"], dataframe["landmark_id"])
        )
        ds = ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        
        if training:
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
            
        ds = ds.batch(Config.BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
    
    def _process_path(self, file_id, label):
        img_path = tf.strings.join([
            str(Config.BASE_PATH),
            file_id[0],
            file_id[1],
            file_id[2],
            file_id + ".jpg"
        ], separator="/")
        
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, Config.IMG_SIZE)
        img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
        
        return img, self.label_encoder(label)
    
    def _augment(self, image, label):
        augmenter = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(Config.DATA_AUGMENTATION["rotation_range"]),
            tf.keras.layers.RandomTranslation(
                Config.DATA_AUGMENTATION["height_shift_range"],
                Config.DATA_AUGMENTATION["width_shift_range"]
            ),
            tf.keras.layers.RandomFlip("horizontal")
        ])
        
        return augmenter(image), label

# model.py
import tensorflow as tf
from config import Config

class LandmarkModel:
    def __init__(self):
        self.model = None
        
    def build(self):
        backbone = getattr(tf.keras.applications, Config.MODEL_PARAMS["backbone"])
        base_model = backbone(
            include_top=False,
            weights="imagenet",
            input_shape=(*Config.IMG_SIZE, 3)
        )
        
        if Config.MODEL_PARAMS["freeze_backbone"]:
            base_model.trainable = False
            
        inputs = tf.keras.Input(shape=(*Config.IMG_SIZE, 3))
        x = base_model(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        if Config.MODEL_PARAMS["use_batch_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)
            
        x = tf.keras.layers.Dense(1024, activation="relu")(x)
        x = tf.keras.layers.Dropout(Config.MODEL_PARAMS["dropout_rate"])(x)
        outputs = tf.keras.layers.Dense(Config.NUM_CLASSES, activation="softmax")(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(Config.LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5)]
        )
        
        return self.model
    
    def train(self, train_ds, val_ds):
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                "best_model.h5",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max"
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=Config.EPOCHS,
            callbacks=callbacks
        )
        
        return history

# inference.py
import tensorflow as tf
import numpy as np
from PIL import Image
from config import Config

class LandmarkPredictor:
    def __init__(self, model_path: str, label_encoder):
        self.model = tf.keras.models.load_model(model_path)
        self.label_encoder = label_encoder
        
    def predict(self, image_path: str, top_k: int = 5):
        img = Image.open(image_path)
        img = img.resize(Config.IMG_SIZE)
        img = np.array(img)
        img = tf.keras.applications.efficientnet_v2.preprocess_input(img)
        img = np.expand_dims(img, axis=0)
        
        predictions = self.model.predict(img)
        top_k_indices = np.argsort(predictions[0])[-top_k:][::-1]
        top_k_probs = predictions[0][top_k_indices]
        
        results = []
        for idx, prob in zip(top_k_indices, top_k_probs):
            landmark_id = self.label_encoder.get_vocabulary()[idx]
            results.append({
                "landmark_id": landmark_id,
                "confidence": float(prob)
            })
            
        return results

# train.py
from data_loader import LandmarkDataLoader
from model import LandmarkModel
from config import Config
import tensorflow as tf
import numpy as np

def main():
    tf.random.set_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    data_loader = LandmarkDataLoader("train.csv")
    train_df, val_df = data_loader.prepare_data()
    
    train_ds = data_loader.create_dataset(train_df, training=True)
    val_ds = data_loader.create_dataset(val_df)
    
    model = LandmarkModel()
    model.build()
    history = model.train(train_ds, val_ds)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
