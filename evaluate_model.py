
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the model from the current directory
model = load_model('best_initial_model.keras')


print(model.output_shape)  # Should be (None, 3) if correctly trained for 3 classes

# Set up the ImageDataGenerator for validation data
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    'dataset/validation',  # Path to your validation data folder
    target_size=(224, 224),       # Image size your model expects
    batch_size=32,
    class_mode='categorical',     # Assuming you have multiple classes (adenocarcinoma, benign, squamous_cell_carcinoma)
    shuffle=False
)

# Evaluate the model on the validation data
loss, accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')








# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report

# # 🔹 Load both models
# original_model_path = "best_model_runtime_0.8078.keras"  # Change this to your actual path
# fine_tuned_model_path = "final_model_runtime_0.7968.keras"  # Change this to your actual path

# original_model = tf.keras.models.load_model(original_model_path)
# fine_tuned_model = tf.keras.models.load_model(fine_tuned_model_path)

# # 🔹 Load Validation Data
# val_data_path = "split_dataset/validation"  # Change this to your dataset path

# datagen = ImageDataGenerator(rescale=1./255)  # Ensure same preprocessing as training

# val_data = datagen.flow_from_directory(
#     val_data_path,
#     target_size=(224, 224),  # Ensure the correct size used in training
#     batch_size=32,
#     class_mode="categorical",
#     shuffle=False
# )

# # 🔹 Get true labels and class names
# true_labels = val_data.classes
# class_names = list(val_data.class_indices.keys())

# # 🔹 Get predictions from both models
# original_preds = original_model.predict(val_data)
# fine_tuned_preds = fine_tuned_model.predict(val_data)

# # Convert predictions to class labels
# original_preds_labels = np.argmax(original_preds, axis=1)
# fine_tuned_preds_labels = np.argmax(fine_tuned_preds, axis=1)

# # 🔹 Generate classification reports
# original_report = classification_report(true_labels, original_preds_labels, target_names=class_names)
# fine_tuned_report = classification_report(true_labels, fine_tuned_preds_labels, target_names=class_names)

# # 🔹 Print reports
# print("🔹 Classification Report - Original Model:")
# print(original_report)

# print("\n🔹 Classification Report - Fine-Tuned Model:")
# print(fine_tuned_report)

# # 🔹 Save reports to files
# with open("original_model_report.txt", "w") as f:
#     f.write(original_report)

# with open("fine_tuned_model_report.txt", "w") as f:
#     f.write(fine_tuned_report)

# print("\n✅ Reports saved as 'original_model_report.txt' and 'fine_tuned_model_report.txt'")










