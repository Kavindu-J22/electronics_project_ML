"""
Main training script for the Device Placement Classifier
"""

import os
import numpy as np
try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from video_classifier import DevicePlacementClassifier

def plot_training_history(history):
    """Plot training history"""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available - skipping plots")
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Initialize classifier
    print("Initializing Device Placement Classifier...")
    classifier = DevicePlacementClassifier(
        img_size=(224, 224),
        frames_per_video=16
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
    print("Loading dataset...")
    X, y = classifier.load_dataset(correctly_worn_dir, incorrectly_worn_dir)
    
    print(f"Dataset loaded successfully!")
    print(f"Total samples: {len(X)}")
    print(f"Correctly worn samples: {np.sum(y == 1)}")
    print(f"Incorrectly worn samples: {np.sum(y == 0)}")
    print(f"Data shape: {X.shape}")
    
    # Create and display model architecture
    print("\nCreating model...")
    model = classifier.create_model()
    print(model.summary())
    
    # Train model
    print("\nStarting training...")
    history = classifier.train(
        X, y,
        validation_split=0.2,
        epochs=25,
        batch_size=2  # Small batch size due to limited data
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    model_path = 'device_placement_model.h5'
    classifier.save_model(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Test on a few samples
    print("\nTesting on sample videos...")
    
    # Test correctly worn videos
    correctly_worn_files = [f for f in os.listdir(correctly_worn_dir) if f.endswith('.mp4')]
    if correctly_worn_files:
        test_video = os.path.join(correctly_worn_dir, correctly_worn_files[0])
        prediction, confidence = classifier.predict_video(test_video)
        print(f"Test video (correctly worn): {correctly_worn_files[0]}")
        print(f"Prediction: {'Correctly worn' if prediction == 1 else 'Incorrectly worn'}")
        print(f"Confidence: {confidence:.3f}")
    
    # Test incorrectly worn videos
    incorrectly_worn_files = [f for f in os.listdir(incorrectly_worn_dir) if f.endswith('.mp4')]
    if incorrectly_worn_files:
        test_video = os.path.join(incorrectly_worn_dir, incorrectly_worn_files[0])
        prediction, confidence = classifier.predict_video(test_video)
        print(f"\nTest video (incorrectly worn): {incorrectly_worn_files[0]}")
        print(f"Prediction: {'Correctly worn' if prediction == 1 else 'Incorrectly worn'}")
        print(f"Confidence: {confidence:.3f}")
    
    print("\nTraining completed successfully!")
    print(f"Model saved as: {model_path}")
    print("Training history plot saved as: training_history.png")
    print("You can now use the trained model for inference on new videos.")

if __name__ == "__main__":
    main()
