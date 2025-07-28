import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from glob import glob
import pandas as pd
from tensorflow.keras import backend as K

# Configuration
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7
NUM_FOLDS = 5
EPOCHS = 50
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
PROJECTION_DIM = 256
NUM_HEADS = 8
TRANSFORMER_LAYERS = 6
MLP_UNITS = 512
DROPOUT_RATE = 0.1
LEARNING_RATE = 1e-4

# Set seeds for reproducibility
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Class mapping for HAM10000
class_mapping = {
    'nv': 0,      # Melanocytic nevi
    'mel': 1,     # Melanoma
    'bkl': 2,     # Benign keratosis-like lesions
    'bcc': 3,     # Basal cell carcinoma
    'akiec': 4,   # Actinic keratoses
    'vasc': 5,    # Vascular lesions
    'df': 6       # Dermatofibroma
}

# Define class_names globally so it's accessible in all functions
class_names = list(class_mapping.keys())

# Function for hair removal via inpainting
def remove_hair(image):
    # Convert to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply blackhat filtering to identify hair-like structures
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create a binary mask of the hair
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Inpaint the original image using the hair mask
    result = cv2.inpaint(image, thresh, 6, cv2.INPAINT_TELEA)
    
    return result

# Albumentations augmentation pipeline
def get_augmentation_transforms():
    return A.Compose([
        A.Rotate(limit=45, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
    ])

# Normalization function based on EfficientNet approach
def normalize_image(image):
    image = image.astype(np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image / 255.0
    image -= mean
    image /= std
    return image

# Dataset preparation function
def preprocess_dataset(data_dir, metadata_file, output_dir="preprocessed_data"):
    """
    Preprocess the HAM10000 dataset and save as NPY files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Read metadata
    df = pd.read_csv(metadata_file)
    
    # Get file paths
    image_files = []
    for ext in ['jpg', 'jpeg', 'png']:
        image_files.extend(glob(os.path.join(data_dir, f"*.{ext}")))
    
    print(f"Found {len(image_files)} images")
    
    images = []
    labels = []
    image_ids = []
    
    for img_path in image_files:
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        row = df[df['image_id'] == img_id]
        
        if len(row) == 0:
            continue
        
        label = row['dx'].values[0]
        if label not in class_mapping:
            continue
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Remove hair
        img = remove_hair(img)
        
        # Resize
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Normalize
        img = normalize_image(img)
        
        images.append(img)
        labels.append(class_mapping[label])
        image_ids.append(img_id)
    
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    ids = np.array(image_ids)
    
    # Save preprocessed data
    np.save(os.path.join(output_dir, 'images.npy'), X)
    np.save(os.path.join(output_dir, 'labels.npy'), y)
    np.save(os.path.join(output_dir, 'image_ids.npy'), ids)
    
    print(f"Preprocessed data saved to {output_dir}")
    print(f"Dataset shape: {X.shape}, Labels shape: {y.shape}")
    
    # Return class distribution for class weighting
    class_counts = np.bincount(y)
    total = len(y)
    class_weights = {}
    for i in range(len(class_counts)):
        class_weights[i] = total / (len(class_counts) * class_counts[i])
    
    return X, y, ids, class_weights

# Custom data generator for NPY files
class NPYDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, augment=False, is_training=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.is_training = is_training
        self.augmentation = get_augmentation_transforms() if augment else None
        self.indices = np.arange(len(self.X))
        if is_training:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_X = self.X[batch_indices].copy()
        batch_y = self.y[batch_indices].copy()
        
        # Apply augmentation if needed
        if self.augment:
            for i in range(len(batch_X)):
                if np.random.rand() < 0.7:  # Apply augmentation with 70% probability
                    aug_data = self.augmentation(image=batch_X[i])
                    batch_X[i] = aug_data['image']
        
        # Convert labels to one-hot
        batch_y = to_categorical(batch_y, num_classes=NUM_CLASSES)
        
        return batch_X, batch_y

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)

# Vision Transformer (ViT) Model
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def create_vit_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Create patches
    patches = layers.Conv2D(
        filters=PROJECTION_DIM,
        kernel_size=PATCH_SIZE,
        strides=PATCH_SIZE,
        padding="valid",
    )(inputs)
    patches = layers.Reshape((NUM_PATCHES, PROJECTION_DIM))(patches)
    
    # Add positional embeddings
    positions = tf.range(start=0, limit=NUM_PATCHES, delta=1)
    position_embedding = layers.Embedding(
        input_dim=NUM_PATCHES, output_dim=PROJECTION_DIM
    )(positions)
    
    # Add class token (for classification)
    batch_size = tf.shape(patches)[0]
    class_token = tf.zeros([batch_size, 1, PROJECTION_DIM])
    patches = layers.Concatenate(axis=1)([class_token, patches])
    
    # Add positional embedding
    positions = tf.range(start=0, limit=NUM_PATCHES + 1, delta=1)
    position_embedding = layers.Embedding(
        input_dim=NUM_PATCHES + 1, output_dim=PROJECTION_DIM
    )(positions)
    x = patches + position_embedding
    
    # Create transformer blocks
    for _ in range(TRANSFORMER_LAYERS):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=NUM_HEADS, key_dim=PROJECTION_DIM // NUM_HEADS, dropout=DROPOUT_RATE
        )(x1, x1)
        
        # Skip connection 1
        x2 = layers.Add()([attention_output, x])
        
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = mlp(x3, hidden_units=[MLP_UNITS, PROJECTION_DIM], dropout_rate=DROPOUT_RATE)
        
        # Skip connection 2
        x = layers.Add()([x3, x2])
    
    # Layer normalization and Global average pooling
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    # Classification head
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# U-Net for Multi-task learning (segmentation + classification)
def conv_block(inputs, filters):
    x = layers.Conv2D(filters, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    return x

def create_unet_multitask_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Contracting path (encoder)
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D()(c1)
    
    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D()(c2)
    
    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D()(c3)
    
    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D()(c4)
    
    # Bottleneck
    bottleneck = conv_block(p4, 1024)
    
    # Expansive path (decoder) for segmentation
    u4 = layers.UpSampling2D()(bottleneck)
    u4 = layers.Concatenate()([u4, c4])
    c4 = conv_block(u4, 512)
    
    u3 = layers.UpSampling2D()(c4)
    u3 = layers.Concatenate()([u3, c3])
    c3 = conv_block(u3, 256)
    
    u2 = layers.UpSampling2D()(c3)
    u2 = layers.Concatenate()([u2, c2])
    c2 = conv_block(u2, 128)
    
    u1 = layers.UpSampling2D()(c2)
    u1 = layers.Concatenate()([u1, c1])
    c1 = conv_block(u1, 64)


    # Continuation of the U-Net multitask model
    # Decoder (segmentation branch)
    u4 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bottleneck)
    u4 = layers.concatenate([u4, c4])
    c5 = conv_block(u4, 512)

    u3 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c5)
    u3 = layers.concatenate([u3, c3])
    c6 = conv_block(u3, 256)

    u2 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c6)
    u2 = layers.concatenate([u2, c2])
    c7 = conv_block(u2, 128)

    u1 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c7)
    u1 = layers.concatenate([u1, c1])
    c8 = conv_block(u1, 64)

    # Segmentation output
    seg_output = layers.Conv2D(1, (1, 1), activation='sigmoid', name='segmentation_output')(c8)

    # Classification head
    # Global pooling on bottleneck or c8
    cls_branch = layers.GlobalAveragePooling2D()(bottleneck)
    cls_branch = layers.Dense(256, activation='relu')(cls_branch)
    cls_branch = layers.Dropout(0.5)(cls_branch)
    cls_output = layers.Dense(NUM_CLASSES, activation='softmax', name='classification_output')(cls_branch)

    # Define the model
    model = models.Model(inputs=inputs, outputs=[seg_output, cls_output])
    return model

    
    # Segmentation output (binary mask)
    segmentation_output = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation')(c1)
    
    # Classification branch from bottleneck
    classification_features = layers.GlobalAveragePooling2D()(bottleneck)
    classification_features = layers.Dense(256, activation='gelu')(classification_features)
    classification_features = layers.Dropout(0.5)(classification_features)
    classification_output = layers.Dense(NUM_CLASSES, activation='softmax', name='classification')(classification_features)
    
    model = models.Model(inputs=inputs, outputs=[segmentation_output, classification_output])
    
    return model

# Metrics and callbacks
def get_callbacks(fold):
    checkpoint_cb = callbacks.ModelCheckpoint(
        f"model_fold_{fold}.h5",
        monitor='val_classification_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max'
    )
    
    early_stopping_cb = callbacks.EarlyStopping(
        monitor='val_classification_accuracy', 
        patience=10, 
        verbose=1, 
        restore_best_weights=True
    )
    
    reduce_lr_cb = callbacks.ReduceLROnPlateau(
        monitor='val_classification_accuracy', 
        factor=0.2, 
        patience=4, 
        min_lr=1e-6, 
        verbose=1
    )
    
    return [checkpoint_cb, early_stopping_cb, reduce_lr_cb]

# Function to calculate Top-K accuracy
def top_k_accuracy(y_true, y_pred, k=2):
    return tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=k)

# Training function for ViT model
def train_vit_model(X, y, class_weights, fold=0):
    print(f"Training ViT model for fold {fold+1}/{NUM_FOLDS}")
    
    # Create train/val split for this fold
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    train_idx, val_idx = list(kf.split(X, y))[fold]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)
    
    # Create data generators
    train_gen = NPYDataGenerator(X_train, y_train, batch_size=BATCH_SIZE, augment=True)
    val_gen = NPYDataGenerator(X_val, y_val, batch_size=BATCH_SIZE, augment=False, is_training=False)
    
    # Create model
    model = create_vit_model()
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', 
                 lambda y_true, y_pred: top_k_accuracy(y_true, y_pred, k=2),
                 lambda y_true, y_pred: top_k_accuracy(y_true, y_pred, k=3)]
    )
    
    # Get callbacks
    callbacks_list = get_callbacks(fold)
    
    # Train model
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluate model
    val_preds = model.predict(X_val)
    val_pred_classes = np.argmax(val_preds, axis=1)
    val_true_classes = np.argmax(y_val_cat, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(val_true_classes, val_pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(val_true_classes, val_pred_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Fold {fold+1}')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_vit_fold_{fold+1}.png')
    
    return model, history, val_preds, val_true_classes

# Training function for U-Net multi-task model
def train_unet_multitask_model(X, y, class_weights, fold=0):
    print(f"Training U-Net multi-task model for fold {fold+1}/{NUM_FOLDS}")
    
    # Create train/val split for this fold
    kf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    train_idx, val_idx = list(kf.split(X, y))[fold]
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Convert labels to categorical
    y_train_cat = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val_cat = to_categorical(y_val, num_classes=NUM_CLASSES)
    
    # For demonstration, we'll create dummy segmentation masks (in a real scenario, you'd have actual masks)
    # This is just a placeholder - in practice, you would use real segmentation masks
    seg_train = np.random.random((len(X_train), IMG_SIZE, IMG_SIZE, 1))
    seg_val = np.random.random((len(X_val), IMG_SIZE, IMG_SIZE, 1))
    
    # Create model
    model = create_unet_multitask_model()
    
    model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss={
        "segmentation_output": "binary_crossentropy",
        "classification_output": "categorical_crossentropy"
    },
    loss_weights={
        "segmentation_output": 1.0,
        "classification_output": 1.0
    },
    metrics={
        "segmentation_output": ["accuracy"],
        "classification_output": ["accuracy"]
    }
)

    # Get callbacks
    callbacks_list = get_callbacks(fold)
    
    # Train model
    history = model.fit(
        X_train,
        {'segmentation': seg_train, 'classification': y_train_cat},
        validation_data=(X_val, {'segmentation': seg_val, 'classification': y_val_cat}),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks_list,
        class_weight={'classification': class_weights},
        verbose=1
    )
    
    # Evaluate model
    val_preds = model.predict(X_val)
    val_pred_classes = np.argmax(val_preds[1], axis=1)  # Classification output is the second output
    val_true_classes = np.argmax(y_val_cat, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(val_true_classes, val_pred_classes, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(val_true_classes, val_pred_classes)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Fold {fold+1}')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(f'confusion_matrix_unet_fold_{fold+1}.png')
    
    return model, history, val_preds, val_true_classes

# Main execution
def main():
    # Using the provided HAM10000 dataset path
    data_dir = "D:\\Major Project\\HAM10000\\HAM10000_images_part_1"  # Updated path to image directory
    metadata_file = "D:\\Major Project\\HAM10000\\HAM10000_metadata.csv"  # Path to metadata file
    
    # Check if paths exist
    if not os.path.exists(data_dir):
        print(f"Error: Image directory not found: {data_dir}")
        print("Please update the path to your image directory.")
        return
        
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found: {metadata_file}")
        print("Please update the path to your metadata file.")
        return
    
    # Preprocess the dataset
    X, y, ids, class_weights = preprocess_dataset(data_dir, metadata_file)
    
    # Class distribution visualization
    class_counts = np.bincount(y)
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_counts)  # class_names is now globally defined
    plt.title('Class Distribution in HAM10000 Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.savefig('class_distribution.png')
    
    # Train ViT model using cross-validation
    vit_val_scores = []
    vit_top2_scores = []
    vit_top3_scores = []
    
    for fold in range(NUM_FOLDS):
        model, history, val_preds, val_true = train_vit_model(X, y, class_weights, fold)
        
        # Save model
        model.save(f"vit_model_fold_{fold+1}.h5")
        
        # Calculate metrics
        val_true_cat = to_categorical(val_true, num_classes=NUM_CLASSES)
        accuracy = np.mean(np.argmax(val_preds, axis=1) == val_true)
        top2_accuracy = np.mean([1 if val_true[i] in np.argsort(val_preds[i])[-2:] else 0 for i in range(len(val_true))])
        top3_accuracy = np.mean([1 if val_true[i] in np.argsort(val_preds[i])[-3:] else 0 for i in range(len(val_true))])
        
        vit_val_scores.append(accuracy)
        vit_top2_scores.append(top2_accuracy)
        vit_top3_scores.append(top3_accuracy)
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(f'ViT Model Accuracy - Fold {fold+1}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'ViT Model Loss - Fold {fold+1}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plt.savefig(f'vit_training_history_fold_{fold+1}.png')
    
    # Train U-Net multi-task model using cross-validation
    unet_val_scores = []
    unet_top2_scores = []
    unet_top3_scores = []
    
    for fold in range(NUM_FOLDS):
        model, history, val_preds, val_true = train_unet_multitask_model(X, y, class_weights, fold)
        
        # Save model
        model.save(f"unet_model_fold_{fold+1}.h5")
        
        # Calculate metrics
        val_true_cat = to_categorical(val_true, num_classes=NUM_CLASSES)
        accuracy = np.mean(np.argmax(val_preds[1], axis=1) == val_true)  # Classification is second output
        top2_accuracy = np.mean([1 if val_true[i] in np.argsort(val_preds[1][i])[-2:] else 0 for i in range(len(val_true))])
        top3_accuracy = np.mean([1 if val_true[i] in np.argsort(val_preds[1][i])[-3:] else 0 for i in range(len(val_true))])
        
        unet_val_scores.append(accuracy)
        unet_top2_scores.append(top2_accuracy)
        unet_top3_scores.append(top3_accuracy)
        
        # Plot training history
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(history.history['classification_accuracy'])
        plt.plot(history.history['val_classification_accuracy'])
        plt.title(f'U-Net Classification Accuracy - Fold {fold+1}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 3, 2)
        plt.plot(history.history['segmentation_accuracy'])
        plt.plot(history.history['val_segmentation_accuracy'])
        plt.title(f'U-Net Segmentation Accuracy - Fold {fold+1}')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 3, 3)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'U-Net Total Loss - Fold {fold+1}')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plt.savefig(f'unet_training_history_fold_{fold+1}.png')
    
    # Print summary of results
    print("\n=== RESULTS SUMMARY ===")
    print("\nVision Transformer (ViT) Model:")
    print(f"Average Accuracy: {np.mean(vit_val_scores):.4f} ± {np.std(vit_val_scores):.4f}")
    print(f"Average Top-2 Accuracy: {np.mean(vit_top2_scores):.4f} ± {np.std(vit_top2_scores):.4f}")
    print(f"Average Top-3 Accuracy: {np.mean(vit_top3_scores):.4f} ± {np.std(vit_top3_scores):.4f}")
    
    print("\nU-Net Multi-task Model:")
    print(f"Average Accuracy: {np.mean(unet_val_scores):.4f} ± {np.std(unet_val_scores):.4f}")
    print(f"Average Top-2 Accuracy: {np.mean(unet_top2_scores):.4f} ± {np.std(unet_top2_scores):.4f}")
    print(f"Average Top-3 Accuracy: {np.mean(unet_top3_scores):.4f} ± {np.std(unet_top3_scores):.4f}")

if __name__ == "__main__":
    main()
