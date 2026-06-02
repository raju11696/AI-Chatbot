import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from transformers import pipeline


# Create Flask app
app = Flask(__name__)


# Folder for uploaded images
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Allowed image file types
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


# Load Hugging Face image classification model
# This is a general image classification model used for a beginner mini project.
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")


# Check uploaded file extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Generate a simple medical chatbot response
def medical_chatbot_response(confidence):
    if confidence >= 0.70:
        result = "Possible Heart Disease Detected"
        explanation = (
            "The AI model found a high-confidence pattern in the uploaded image. "
            "This may indicate a possible heart-related abnormality."
        )
        precautions = [
            "Consult a cardiologist as soon as possible.",
            "Avoid heavy physical activity until medical advice is received.",
            "Monitor chest pain, breathing difficulty, and blood pressure.",
        ]
        recommendations = [
            "Get professional tests such as ECG, Echo, or MRI.",
            "Follow a heart-healthy diet.",
            "Take prescribed medicines only under doctor guidance.",
        ]
    elif confidence >= 0.40:
        result = "Possible Arrhythmia Detected"
        explanation = (
            "The AI model found a medium-confidence pattern. "
            "This may suggest possible irregular heart rhythm or mild abnormality."
        )
        precautions = [
            "Schedule a routine heart checkup.",
            "Reduce stress, smoking, caffeine, and alcohol.",
            "Track heartbeat changes or dizziness.",
        ]
        recommendations = [
            "Consider an ECG test if symptoms continue.",
            "Maintain regular sleep and exercise habits.",
            "Discuss symptoms with a medical professional.",
        ]
    else:
        result = "No Major Abnormality Detected"
        explanation = (
            "The AI model found a low-confidence abnormal pattern. "
            "No major abnormality is detected by this simple project."
        )
        precautions = [
            "Continue regular health checkups.",
            "Seek help if chest pain or breathlessness occurs.",
            "Do not depend only on AI for medical decisions.",
        ]
        recommendations = [
            "Exercise regularly.",
            "Eat a balanced diet.",
            "Maintain healthy blood pressure and cholesterol levels.",
        ]

    return result, explanation, precautions, recommendations


# Home page route
@app.route("/")
def home():
    return render_template("home.html")


# Route for showing uploaded images on the result page
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("home.html", error="Please upload an image.")

    file = request.files["image"]

    if file.filename == "":
        return render_template("home.html", error="Please choose an image file.")

    if not allowed_file(file.filename):
        return render_template("home.html", error="Only JPG, JPEG, and PNG files are allowed.")

    # Save uploaded image
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Run Hugging Face image classification
    prediction = classifier(filepath)[0]
    label = prediction["label"]
    confidence = prediction["score"]

    # Generate chatbot-style medical response
    result, explanation, precautions, recommendations = medical_chatbot_response(confidence)

    return render_template(
        "result.html",
        filename=filename,
        label=label,
        confidence=round(confidence * 100, 2),
        result=result,
        explanation=explanation,
        precautions=precautions,
        recommendations=recommendations,
    )


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
