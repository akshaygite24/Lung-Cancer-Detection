import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Enable mixed precision to speed up training
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Enable XLA compilation for performance boost
tf.config.optimizer.set_jit(True)

# Mount Google Drive to save model
from google.colab import drive
drive.mount('/content/drive')

# Set parameters
img_height, img_width = 224, 224
batch_size = 64  # Larger batch size for faster training (if GPU allows)
epochs = 20  # Limited to 20 with early stopping

# Dataset paths
base_dir = "/content/dataset"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

# Load InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Fine-tune more layers
for layer in base_model.layers[:80]:
    layer.trainable = False  # Freeze first 80 layers
for layer in base_model.layers[80:]:
    layer.trainable = True

# Custom classification layers
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(512, activation='relu'),
    Dropout(0.3),
    Dense(5, activation='softmax', dtype='float32')  # Ensure float32 for stability
])

# Compile with Adam optimizer and reduced learning rate
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation (Optimized for speed)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    validation_dir, target_size=(img_height, img_width),
    batch_size=batch_size, class_mode='categorical', shuffle=False
)

# Compute class weights for balancing
labels = train_generator.classes
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

print("Class Indices:", train_generator.class_indices)

# Callbacks for stable training
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=2, verbose=1, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Save model in runtime & Google Drive with accuracy in filename
checkpoint_drive = ModelCheckpoint(
    filepath='/content/drive/MyDrive/best_model_{val_accuracy:.4f}.keras',
    save_best_only=True, monitor='val_accuracy', mode='max'
)

checkpoint_runtime = ModelCheckpoint(
    filepath='/content/best_model_runtime_{val_accuracy:.4f}.keras',
    save_best_only=True, monitor='val_accuracy', mode='max'
)

# Train model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    class_weight=class_weights_dict,
    callbacks=[lr_scheduler, early_stopping, checkpoint_drive, checkpoint_runtime]
)

# Evaluate model
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Save final model with accuracy in filename
final_model_path_drive = f"/content/drive/MyDrive/final_model_{val_accuracy:.4f}.keras"
final_model_path_runtime = f"/content/final_model_runtime_{val_accuracy:.4f}.keras"

model.save(final_model_path_drive)
model.save(final_model_path_runtime)

print(f"Final model saved: {final_model_path_drive} and {final_model_path_runtime}")

# Get predictions
y_true = validation_generator.classes
y_pred = model.predict(validation_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
class_names = ["Adenocarcinoma", "Benign", "Healthy Lung", "Random Images", "Squamous Cell Carcinoma"]
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
