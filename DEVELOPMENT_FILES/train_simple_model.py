"""
Training script for the Simple Device Placement Classifier
"""

import os
import numpy as np
from simple_classifier import SimpleDeviceClassifier

def main():
    # Initialize classifier
    print("Initializing Simple Device Placement Classifier...")
    classifier = SimpleDeviceClassifier(
        img_size=(224, 224),
        frames_per_video=8  # Fewer frames for faster processing
    )
    
    # Data directories
    correctly_worn_dir = 'correctly worn'
    incorrectly_worn_dir = 'incorrectly worn'
    
    # Check if directories exist
    if not os.path.exists(correctly_worn_dir):
        raise FileNotFoundError(f"Directory '{correctly_worn_dir}' not found!")
    if not os.path.exists(incorrectly_worn_dir):
        raise FileNotFoundError(f"Directory '{incorrectly_worn_dir}' not found!")
    
    # Load dataset
    print("Loading dataset and extracting features...")
    X, y = classifier.load_dataset(correctly_worn_dir, incorrectly_worn_dir)
    
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(X)}")
    print(f"Correctly worn samples: {np.sum(y == 1)}")
    print(f"Incorrectly worn samples: {np.sum(y == 0)}")
    print(f"Feature dimension: {X.shape[1]}")
    
    # Train Random Forest model
    print("\n" + "="*50)
    print("Training Random Forest Model")
    print("="*50)
    rf_accuracy = classifier.train(X, y, model_type='random_forest', test_size=0.3)
    
    # Save Random Forest model
    rf_model_path = 'simple_rf_model.pkl'
    classifier.save_model(rf_model_path)
    
    # Train SVM model
    print("\n" + "="*50)
    print("Training SVM Model")
    print("="*50)
    svm_classifier = SimpleDeviceClassifier(
        img_size=(224, 224),
        frames_per_video=8
    )
    svm_accuracy = svm_classifier.train(X, y, model_type='svm', test_size=0.3)
    
    # Save SVM model
    svm_model_path = 'simple_svm_model.pkl'
    svm_classifier.save_model(svm_model_path)
    
    # Compare models
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    print(f"Random Forest Accuracy: {rf_accuracy:.3f}")
    print(f"SVM Accuracy: {svm_accuracy:.3f}")
    
    # Choose best model
    if rf_accuracy >= svm_accuracy:
        best_model = 'Random Forest'
        best_accuracy = rf_accuracy
        best_path = rf_model_path
        best_classifier = classifier
    else:
        best_model = 'SVM'
        best_accuracy = svm_accuracy
        best_path = svm_model_path
        best_classifier = svm_classifier
    
    print(f"Best model: {best_model} (Accuracy: {best_accuracy:.3f})")
    
    # Test on sample videos
    print("\n" + "="*50)
    print("TESTING ON SAMPLE VIDEOS")
    print("="*50)
    
    # Test correctly worn videos
    correctly_worn_files = [f for f in os.listdir(correctly_worn_dir) if f.endswith('.mp4')]
    if correctly_worn_files:
        test_video = os.path.join(correctly_worn_dir, correctly_worn_files[0])
        prediction, confidence = best_classifier.predict_video(test_video)
        print(f"Test video (correctly worn): {correctly_worn_files[0]}")
        print(f"Prediction: {'✅ Correctly worn' if prediction == 1 else '❌ Incorrectly worn'}")
        print(f"Confidence: {confidence:.3f}")
    
    # Test incorrectly worn videos
    incorrectly_worn_files = [f for f in os.listdir(incorrectly_worn_dir) if f.endswith('.mp4')]
    if incorrectly_worn_files:
        test_video = os.path.join(incorrectly_worn_dir, incorrectly_worn_files[0])
        prediction, confidence = best_classifier.predict_video(test_video)
        print(f"\nTest video (incorrectly worn): {incorrectly_worn_files[0]}")
        print(f"Prediction: {'✅ Correctly worn' if prediction == 1 else '❌ Incorrectly worn'}")
        print(f"Confidence: {confidence:.3f}")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Best model saved as: {best_path}")
    print("Confusion matrix saved as: simple_confusion_matrix.png")
    print("\nYou can now use the trained model for inference on new videos.")
    print(f"Example: python predict_simple.py 'path/to/video.mp4'")

if __name__ == "__main__":
    main()
