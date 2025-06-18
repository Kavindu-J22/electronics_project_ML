"""
Quick training script - minimal version
"""

import os
import numpy as np
from simple_classifier import SimpleDeviceClassifier

def main():
    print("Quick training...")
    
    # Initialize classifier
    classifier = SimpleDeviceClassifier(img_size=(128, 128), frames_per_video=4)
    
    # Load dataset
    X, y = classifier.load_dataset('correctly worn', 'incorrectly worn')
    print(f"Loaded {len(X)} samples with {X.shape[1]} features")
    
    # Train model
    accuracy = classifier.train(X, y, model_type='random_forest', test_size=0.3)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Save model
    classifier.save_model('device_classifier_model.pkl')
    
    # Test prediction
    test_video = 'correctly worn/16.mp4'
    if os.path.exists(test_video):
        pred, conf = classifier.predict_video(test_video)
        print(f"Test: {test_video} -> {'Correct' if pred == 1 else 'Incorrect'} ({conf:.3f})")
    
    print("Done!")

if __name__ == "__main__":
    main()
