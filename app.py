from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
from PIL import Image
import uvicorn
import os
from fastapi.middleware.cors import CORSMiddleware

# Load the updated trained model
model_path = "best_initial_model.keras"  # Ensure this is the correct path
if os.path.isfile(model_path):
    print("Updated model file exists! Loading...")
    model = tf.keras.models.load_model(model_path)
else:
    raise FileNotFoundError("Updated model file not found! Please check the path.")

# Define FastAPI app
app = FastAPI()

# Updated class labels (only three classes)
CLASS_NAMES = ["Adenocarcinoma", "Benign", "Squamous Cell Carcinoma"]

# Function to preprocess images
def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to provide feedback based on prediction and confidence
def get_feedback(prediction, confidence):
    feedback_messages = {
        "Adenocarcinoma": [
            "The model strongly suggests adenocarcinoma. Please consult a doctor immediately for further diagnosis.",
            "There's a moderate likelihood of adenocarcinoma. It's recommended to seek medical advice for confirmation.",
            "The possibility of adenocarcinoma exists, but it's uncertain. A follow-up with a specialist is advised."
        ],
        "Benign": [
            "The model suggests a benign condition with high confidence. However, if symptoms persist, consult a doctor for reassurance.",
            "The model leans towards a benign condition, but medical consultation is recommended to confirm.",
            "The results indicate a benign condition, but there is some uncertainty. Further medical testing may be helpful."
        ],
        "Squamous Cell Carcinoma": [
            "The model strongly suggests squamous cell carcinoma. Please consult a doctor immediately for further evaluation.",
            "There's a moderate likelihood of squamous cell carcinoma. We recommend seeking professional medical advice.",
            "The results indicate a possibility of squamous cell carcinoma, but it's uncertain. Further medical testing is advised."
        ]
    }
    
    if confidence >= 0.80:
        return feedback_messages[prediction][0]
    elif confidence >= 0.50:
        return feedback_messages[prediction][1]
    else:
        return feedback_messages[prediction][2]

# CORS Middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allowing all origins to avoid CORS errors
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# API Route for prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        img = preprocess_image(file.file)
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        confidence = np.max(prediction)

        feedback = get_feedback(CLASS_NAMES[class_index], confidence)

        return {
            "prediction": CLASS_NAMES[class_index],
            "confidence": float(confidence),
            "feedback": feedback
        }
    except Exception as e:
        return {"error": str(e)}

# Run API locally
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)