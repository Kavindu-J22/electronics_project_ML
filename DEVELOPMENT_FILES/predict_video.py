"""
Inference script for predicting device placement from video files
"""

import sys
import os
from video_classifier import DevicePlacementClassifier

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_video.py <video_path>")
        print("Example: python predict_video.py 'correctly worn/16.mp4'")
        return
    
    video_path = sys.argv[1]
    model_path = 'device_placement_model.h5'
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first by running: python train_model.py")
        return
    
    # Initialize classifier and load model
    print("Loading trained model...")
    classifier = DevicePlacementClassifier()
    classifier.load_model(model_path)
    
    # Make prediction
    print(f"Analyzing video: {video_path}")
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
