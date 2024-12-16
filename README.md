# Humor Detection App

## Overview
The Humor Detection App is a machine learning-based project that classifies user-input text as either humorous or non-humorous. The project utilizes an NLP model for text classification, with the backend implemented using Flask and the frontend as an Android application.

## Key Features
- **Text Classification:** Predicts whether a given sentence is humorous or not.
- **Interactive UI:** Android app for seamless user interaction.
- **Efficient Backend:** Flask API for processing user inputs and returning predictions.

## Technologies Used
- **Frontend:** Android (Kotlin, Capacitor)
- **Backend:** Flask
- **Machine Learning Model:** NLP model fine-tuned for humor detection
- **Dataset:** Contains text samples labeled as 'humorous' or 'non-humorous'.

## Application Flow
1. **User Input:** The user enters a sentence in the Android app.
2. **Request:** The app sends the text to the Flask API.
3. **Prediction:** The Flask API processes the text using the NLP model and returns the prediction.
4. **Result:** The app displays whether the input is humorous or non-humorous.

## Project Structure
- **Frontend (Android App):**
  - UI components and input handling.
  - API requests to the backend.
- **Backend (Flask API):**
  - Pre-trained humor detection model in `.pth` format.
  - API endpoint for handling input and returning predictions.
- **NLP Model:**
  - Pretrained and fine-tuned T5 model for text classification.

## Usage
1. Launch the Android app.
2. Enter a sentence in the input field.
3. Tap "Check" to classify the input text.
4. View the result displayed as "Humorous" or "Non-Humorous."

## Future Improvements
- Add support for additional languages.
- Enhance the dataset to improve prediction accuracy.
- Deploy the backend on a cloud platform for wider accessibility.

---


