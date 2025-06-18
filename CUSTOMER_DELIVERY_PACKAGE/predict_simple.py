"""
Simple inference script for predicting device placement from video files
"""

import sys
import os
from simple_classifier import SimpleDeviceClassifier

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_simple.py <video_path>")
        print("Example: python predict_simple.py 'correctly worn/16.mp4'")
        return
    
    video_path = sys.argv[1]
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    
    # Look for available models
    model_paths = ['device_classifier_model.pkl', 'simple_rf_model.pkl', 'simple_svm_model.pkl']
    model_path = None

    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("Error: No trained model found!")
        print("Available models to look for:")
        for path in model_paths:
            print(f"  - {path}")
        print("Please train a model first by running: python train_simple_model.py")
        return
    
    # Initialize classifier and load model
    print(f"Loading trained model: {model_path}")
    classifier = SimpleDeviceClassifier()
    classifier.load_model(model_path)
    
    # Make prediction
    print(f"Analyzing video: {video_path}")
    print("Extracting features...")
    
    try:
        prediction, confidence = classifier.predict_video(video_path)
        
        # Display results
        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Video: {video_path}")
        print(f"Prediction: {'✅ CORRECTLY WORN' if prediction == 1 else '❌ INCORRECTLY WORN'}")
        print(f"Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        
        if confidence > 0.8:
            print("🔥 High confidence prediction")
        elif confidence > 0.6:
            print("⚠️  Medium confidence prediction")
        else:
            print("⚠️  Low confidence prediction - consider manual review")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
