import tensorflow as tf
import numpy as np
import json
import cv2
import os

# ============================================================================
# CUSTOM LOSS FUNCTION (SAME AS TRAINING NOTEBOOK)
# ============================================================================

def weighted_binary_crossentropy(y_true, y_pred, class_weights):
    """
    Weighted Binary Cross-Entropy Loss for multi-label classification
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    weights = tf.constant(class_weights, dtype=tf.float32)
    weighted_bce = bce * weights

    return tf.reduce_mean(weighted_bce)


# ============================================================================
# CHEST X-RAY MODEL LOADER CLASS
# ============================================================================

class ChestXRayModel:

    DISEASE_CLASSES = [
        "Atelectasis", "Consolidation", "Infiltration", "Pneumothorax",
        "Edema", "Emphysema", "Fibrosis", "Effusion", "Pneumonia",
        "Pleural_Thickening", "Cardiomegaly", "Nodule", "Mass", "Hernia"
    ]

    def __init__(self, model_path, metadata_path, thresholds_path, class_weights_path=None):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.thresholds_path = thresholds_path
        self.class_weights_path = class_weights_path

        self.model = None
        self.class_names = None
        self.thresholds = None
        self.class_weights = None

        self.load_all()

    # ====================================================================
    # LOAD EVERYTHING
    # ====================================================================
    def load_all(self):
        print("Loading model components...")
        self.load_model()
        self.load_metadata()
        self.load_thresholds()
        if self.class_weights_path and os.path.exists(self.class_weights_path):
            self.load_class_weights()
        print("✓ All model components loaded successfully!")

    # ====================================================================
    # LOAD MODEL
    # ====================================================================
    def load_model(self):

        def loss_wrapper(y_true, y_pred):
            weights = np.ones(len(self.DISEASE_CLASSES), dtype=np.float32)
            if self.class_weights is not None:
                weights = self.class_weights
            return weighted_binary_crossentropy(y_true, y_pred, weights)

        try:
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    "loss": loss_wrapper,
                    "weighted_binary_crossentropy": weighted_binary_crossentropy
                },
                compile=False
            )
            print(f"✓ Model loaded from: {self.model_path}")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    # ====================================================================
    # LOAD METADATA
    # ====================================================================
    def load_metadata(self):
        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)
                self.class_names = metadata.get("class_names", self.DISEASE_CLASSES)

            print(f"✓ Metadata loaded: {len(self.class_names)} classes")

        except Exception as e:
            print(f"⚠️ Warning: Could not load metadata: {e}")
            self.class_names = self.DISEASE_CLASSES

    # ====================================================================
    # LOAD THRESHOLDS
    # ====================================================================
    def load_thresholds(self):
        try:
            self.thresholds = np.load(self.thresholds_path)
            print(f"✓ Thresholds loaded: {len(self.thresholds)} values")

        except Exception as e:
            print(f"⚠️ Warning: Could not load thresholds, using default 0.5")
            self.thresholds = np.ones(len(self.DISEASE_CLASSES)) * 0.5

    # ====================================================================
    # LOAD CLASS WEIGHTS
    # ====================================================================
    def load_class_weights(self):
        try:
            self.class_weights = np.load(self.class_weights_path)
            print("✓ Class weights loaded")

        except Exception as e:
            print(f"⚠️ Warning: Could not load class weights: {e}")

    # ====================================================================
    # IMAGE PREPROCESS
    # ====================================================================
    def preprocess_image(self, image_path_or_array):

        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Could not load image: {image_path_or_array}")
        else:
            image = image_path_or_array

        img = cv2.resize(image, (224, 224))

        # Fix channel format
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    # ====================================================================
    # FIXED PREDICT METHOD (FINAL VERSION)
    # ====================================================================
    def predict(self, image_path_or_array, return_all_scores=False):
        """Make predictions on a chest X-ray image"""

        processed_img = self.preprocess_image(image_path_or_array)

        # Raw predictions (shape: (1, 14)) → take index 0
        raw_predictions = self.model.predict(processed_img, verbose=0)[0]

        # Build full score dictionary
        all_scores = {}
        for i in range(len(self.class_names)):
            all_scores[self.class_names[i]] = float(raw_predictions[i])

        # Apply thresholds
        results = []
        for i, class_name in enumerate(self.class_names):
            score = float(raw_predictions[i])
            threshold = float(self.thresholds[i])

            if score >= threshold:
                results.append({
                    "disease": class_name,
                    "confidence": score,
                    "threshold": threshold,
                    "detected": True
                })

        # Sort by highest confidence
        results = sorted(results, key=lambda x: x["confidence"], reverse=True)

        if return_all_scores:
            return results, raw_predictions, all_scores

        return results, raw_predictions


# ============================================================================
# DIRECT EXECUTION TEST
# ============================================================================
if __name__ == "__main__":
    model = ChestXRayModel(
        model_path="final_densenet121_model.h5",
        metadata_path="model_metadata.json",
        thresholds_path="optimal_thresholds.npy",
        class_weights_path="class_weights.npy"
    )

    try:
        detected, scores = model.predict("path_to_xray.png")
        print("\nDetected Diseases:")
        for d in detected:
            print(f" - {d['disease']}: {d['confidence']:.3f}")
    except Exception as e:
        print("Error:", e)
