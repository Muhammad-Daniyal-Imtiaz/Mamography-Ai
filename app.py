!pip install kagglehub
import kagglehub
import os
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Step 1: Correct Dataset Download and Extraction
!pip install kagglehub
import kagglehub
import os
import shutil

# Download dataset - this returns the directory path where files are downloaded
download_path = kagglehub.dataset_download("rabiaedaylmaz/vindr-mammo-processed-512-itu")
print("Download path:", download_path)

# Create target directory
target_dir = '/content/vindr-mammo'
os.makedirs(target_dir, exist_ok=True)

# Copy all files from download directory to our target directory
for item in os.listdir(download_path):
    src = os.path.join(download_path, item)
    dst = os.path.join(target_dir, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print("Files copied to:", target_dir)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. First, let's properly examine the directory structure
base_dir = '/content/vindr-mammo'
image_dir = os.path.join(base_dir, 'images_processed_cv2_dicomsdl_512')

print("\nFull directory structure:")
!find {base_dir} -maxdepth 3 -type d | sort

print("\nFiles in image directory:")
!ls -lh {image_dir} | head -10

# 2. Create a proper file listing with full paths
all_items = os.listdir(image_dir)
print(f"\nTotal items in image directory: {len(all_items)}")

# Separate files and directories
dirs = [d for d in all_items if os.path.isdir(os.path.join(image_dir, d))]
files = [f for f in all_items if os.path.isfile(os.path.join(image_dir, f))]

print(f"Subdirectories found: {len(dirs)}")
print(f"Files found: {len(files)}")

# 3. If we found subdirectories, they likely contain the actual images
if dirs:
    print("\nExamining first subdirectory...")
    first_subdir = os.path.join(image_dir, dirs[0])
    subdir_files = os.listdir(first_subdir)
    print(f"Files in first subdirectory: {len(subdir_files)}")
    print("First 5 files:", subdir_files[:5])

    # Check if these are image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.dcm')
    image_files = [f for f in subdir_files if f.lower().endswith(image_extensions)]
    print(f"Image files found: {len(image_files)}")

    if image_files:
        # Create dataframe from these images
        df = pd.DataFrame({
            'patient_dir': dirs[0],
            'image_file': image_files[:1000]  # Limit to first 1000 for demo
        })

        # Create full paths
        df['image_path'] = df.apply(
            lambda x: os.path.join(image_dir, x['patient_dir'], x['image_file']),
            axis=1
        )

        # Simulate BIRADS data for demonstration
        import numpy as np
        np.random.seed(42)
        df['breast_birads'] = np.random.choice(
            ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4', 'BI-RADS 5'],
            size=len(df),
            p=[0.2, 0.3, 0.25, 0.15, 0.1]
        )

        # 4. Create visualization
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='breast_birads', order=sorted(df['breast_birads'].unique()))
        plt.title('Simulated BIRADS Distribution (Demo)', fontsize=14)
        plt.xlabel('BIRADS Category', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.show()

        print("\nSample image paths:")
        print(df[['image_file', 'breast_birads']].head())
    else:
        print("No image files found in subdirectories")
else:
    print("No subdirectories found - please check dataset structure")

!pip install pydicom

import cv2
from PIL import Image
import pydicom
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 1. Create a proper dataframe with metadata (if available)
# If metadata CSV exists, load it. Otherwise create a simulated one.
metadata_path = os.path.join(base_dir, 'metadata.csv')
if os.path.exists(metadata_path):
    df_meta = pd.read_csv(metadata_path)
else:
    # Simulate metadata for demonstration
    patient_ids = dirs[:1000]  # Using first 1000 patients
    df_meta = pd.DataFrame({
        'patient_id': patient_ids,
        'age': np.random.randint(30, 80, size=len(patient_ids)),
        'biopsy_result': np.random.choice(['Benign', 'Malignant'], size=len(patient_ids), p=[0.7, 0.3]),
        'density': np.random.choice(['A', 'B', 'C', 'D'], size=len(patient_ids))
    })

# 2. Image preprocessing function
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        if image_path.lower().endswith('.dcm'):
            # DICOM file handling
            ds = pydicom.dcmread(image_path)
            img = ds.pixel_array
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            # Regular image file
            img = cv2.imread(image_path)

        # Resize and normalize
        img = cv2.resize(img, target_size)
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# 3. Prepare dataset
def prepare_dataset(df_meta, image_dir, sample_size=1000):
    data = []
    labels = []

    for idx, row in df_meta.iterrows():
        patient_id = row['patient_id']
        patient_dir = os.path.join(image_dir, patient_id)

        if os.path.exists(patient_dir):
            images = [f for f in os.listdir(patient_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm'))]

            # Take first image for each patient for simplicity
            if images:
                img_path = os.path.join(patient_dir, images[0])
                img = preprocess_image(img_path)

                if img is not None:
                    data.append(img)
                    labels.append(row['biopsy_result'])

        if len(data) >= sample_size:
            break

    # Convert to numpy arrays
    X = np.array(data)
    y = np.array(labels)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)  # 0 for Benign, 1 for Malignant

    return X, y, le

# Prepare the dataset
X, y, label_encoder = prepare_dataset(df_meta, image_dir)
print(f"Dataset prepared with {X.shape[0]} samples, image shape: {X.shape[1:]}")

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create data generators
train_generator = train_datagen.flow(
    X_train, y_train,
    batch_size=32,
    shuffle=True
)

test_datagen = ImageDataGenerator()  # No augmentation for validation/test
test_generator = test_datagen.flow(
    X_test, y_test,
    batch_size=32,
    shuffle=False
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def build_baseline_cnn(input_shape=(224, 224, 3)):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    return model

# Build and show model summary
baseline_model = build_baseline_cnn()
baseline_model.summary()

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_auc',
    patience=10,
    mode='max',
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    'best_baseline_model.h5',
    monitor='val_auc',
    save_best_only=True,
    mode='max'
)

# Train the model
history = baseline_model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    callbacks=[early_stopping, checkpoint]
)

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['auc'], label='Train AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_model(model, test_generator):
    # Get true labels and predictions
    y_true = test_generator.labels if hasattr(test_generator, 'labels') else None

    # If no labels attribute, get them from the generator
    if y_true is None:
        test_generator.reset()
        y_true = []
        for i in range(len(test_generator)):
            _, batch_labels = test_generator[i]
            y_true.extend(batch_labels)
        y_true = np.array(y_true)

    # Get predictions
    test_generator.reset()
    y_pred = model.predict(test_generator)
    y_pred_classes = (y_pred > 0.5).astype(int)

    # Classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=['Benign', 'Malignant']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Evaluate the baseline model
evaluate_model(baseline_model, test_generator)

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model

def build_transfer_model(input_shape=(224, 224, 3)):
    # Create base model
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'  # Add global average pooling directly in base model
    )

    # Freeze base model layers
    base_model.trainable = False

    # Create new model on top
    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model

# Usage
transfer_model = build_transfer_model()
transfer_model.summary()

def build_multimodal_model(image_shape=(224, 224, 3), num_clinical_features=5):
    # Image processing branch
    image_input = layers.Input(shape=image_shape, name='image_input')
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=image_shape,
        pooling='avg'
    )
    base_model.trainable = False
    image_features = base_model(image_input)
    image_branch = layers.Dense(128, activation='relu')(image_features)

    # Clinical data processing branch
    clinical_input = layers.Input(shape=(num_clinical_features,), name='clinical_input')
    clinical_branch = layers.Dense(64, activation='relu')(clinical_input)
    clinical_branch = layers.Dense(64, activation='relu')(clinical_branch)

    # Combined processing
    combined = layers.Concatenate()([image_branch, clinical_branch])
    combined = layers.Dense(128, activation='relu')(combined)
    combined = layers.Dropout(0.5)(combined)
    output = layers.Dense(1, activation='sigmoid')(combined)

    # Create model
    model = Model(
        inputs=[image_input, clinical_input],
        outputs=output,
        name='multimodal_model'
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )

    return model

# Usage
multimodal_model = build_multimodal_model()
multimodal_model.summary()

# 1. First clean install with compatible versions
!pip uninstall -y tensorflow keras tensorflow_addons vit-keras
!pip install tensorflow==2.12.0
!pip install tensorflow-addons==0.21.0
!pip install vit-keras==0.1.2

# 2. Restart runtime (Essential!)
from IPython.display import clear_output
clear_output()
print("Please go to Runtime -> Restart runtime now, then re-run this cell")

# 3. Verify installations
import tensorflow as tf
import tensorflow_addons as tfa
from vit_keras import vit

print(f"TensorFlow: {tf.__version__}")
print(f"TFA: {tfa.__version__}")
print(f"ViT: vit_keras.__version__")

# 4. Build ViT model with proper imports
from tensorflow.keras import layers, Model

def build_vit_model(input_shape=(224, 224, 3)):
    vit_model = vit.vit_b16(
        image_size=input_shape[0],
        activation=None,
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )
    vit_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = vit_model(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# 5. Test the model
try:
    vit_model = build_vit_model()
    vit_model.summary()
    print("\nSuccess! ViT model built correctly.")
except Exception as e:
    print(f"Error: {str(e)}")
    print("\nIf you see errors:")
    print("1. Go to Runtime -> Restart runtime")
    print("2. Run all cells again")
    print("3. Check versions with: !pip list | grep 'tensorflow\\|keras\\|addons'")

import numpy as np
from tensorflow.keras import layers, Model
import tensorflow as tf

# 1. First define all individual model builders
def build_cnn_model(input_shape=(224, 224, 3)):
    """Basic CNN model for ensemble"""
    model = tf.keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

def build_transfer_model(input_shape=(224, 224, 3)):
    """EfficientNet transfer learning model"""
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling='avg'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# 2. Keep your existing ViT model builder
def build_vit_model(input_shape=(224, 224, 3)):
    vit_model = vit.vit_b16(
        image_size=input_shape[0],
        activation=None,
        pretrained=True,
        include_top=False,
        pretrained_top=False
    )
    vit_model.trainable = False

    model = tf.keras.Sequential([
        vit_model,
        layers.LayerNormalization(),
        layers.Dense(128, activation='gelu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model

# 3. Ensemble class (unchanged from your working version)
class EnsembleModel:
    def __init__(self, models):
        self.models = models
        self.ensemble = self._build_ensemble()

    def _build_ensemble(self):
        model_input = layers.Input(shape=self.models[0].input_shape[1:])
        outputs = [model(model_input) for model in self.models]
        ensemble_output = layers.Average()(outputs)

        ensemble = Model(
            inputs=model_input,
            outputs=ensemble_output,
            name='ensemble_model'
        )

        ensemble.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return ensemble

    def predict(self, x):
        return self.ensemble.predict(x)

    def evaluate(self, x, y):
        return self.ensemble.evaluate(x, y)

# 4. Example usage
if __name__ == "__main__":
    # Initialize all models
    print("Building models...")
    vit_model = build_vit_model()
    cnn_model = build_cnn_model()
    effnet_model = build_transfer_model()

    # Create ensemble
    print("Creating ensemble...")
    ensemble = EnsembleModel([vit_model, cnn_model, effnet_model])

    # Display summary
    print("\nEnsemble Model Summary:")
    ensemble.ensemble.summary()

    print("\nEnsemble created successfully!")

import numpy as np
from IPython.display import display, Image
import ipywidgets as widgets
from PIL import Image as PILImage
import matplotlib.pyplot as plt

def load_and_predict(model, image_path, target_size=(224, 224)):
    """Load, preprocess and predict on single image"""
    # Load image
    img = PILImage.open(image_path)
    display(img)  # Show original image

    # Convert to RGB if grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Preprocess
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]
    return prediction

def interactive_predictor(model):
    """Create interactive prediction widget"""
    uploader = widgets.FileUpload(
        accept='.jpg,.jpeg,.png',
        multiple=False,
        description='Select Image'
    )

    output = widgets.Output()

    def on_upload_change(change):
        with output:
            output.clear_output()
            if not uploader.value:
                return

            # Get uploaded file
            uploaded = next(iter(uploader.value.values()))
            img_path = 'temp_img.jpg'
            with open(img_path, 'wb') as f:
                f.write(uploaded['content'])

            try:
                # Predict and display
                pred = load_and_predict(model, img_path)
                plt.figure(figsize=(4, 1))
                plt.barh(['Malignant', 'Benign'], [pred, 1-pred], color=['red', 'green'])
                plt.xlim(0, 1)
                plt.title(f"Prediction: {pred:.2%} malignant")
                plt.show()
            except Exception as e:
                print(f"Error: {str(e)}")
                print("Please ensure you upload a valid RGB image")

    uploader.observe(on_upload_change, names='value')
    display(uploader, output)

# Usage:
print("Select an image for prediction:")
interactive_predictor(ensemble.ensemble)  # Use your ensemble model
