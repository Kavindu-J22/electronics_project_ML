"""
Stroke Rehabilitation Device Placement Classifier
This module contains the main video classification pipeline for detecting
whether a rehabilitation device is worn correctly or incorrectly.
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Use simple_classifier.py instead.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        if desc:
            print(f"{desc}...")
        return iterable

import pickle

class DevicePlacementClassifier:
    def __init__(self, img_size=(224, 224), frames_per_video=16):
        """
        Initialize the classifier
        
        Args:
            img_size: Target image size for frames (width, height)
            frames_per_video: Number of frames to extract from each video
        """
        self.img_size = img_size
        self.frames_per_video = frames_per_video
        self.model = None
        self.class_names = ['incorrectly_worn', 'correctly_worn']
        
    def extract_frames(self, video_path, max_frames=None):
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frames as numpy arrays
        """
        if max_frames is None:
            max_frames = self.frames_per_video
            
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return frames
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to extract evenly distributed frames
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Resize frame
                frame = cv2.resize(frame, self.img_size)
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Normalize pixel values
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        
        cap.release()
        return frames
    
    def load_dataset(self, correctly_worn_dir, incorrectly_worn_dir):
        """
        Load and preprocess the video dataset
        
        Args:
            correctly_worn_dir: Directory containing correctly worn videos
            incorrectly_worn_dir: Directory containing incorrectly worn videos
            
        Returns:
            X: Feature array (videos as frame sequences)
            y: Labels array
        """
        X = []
        y = []
        
        print("Loading correctly worn videos...")
        for filename in tqdm(os.listdir(correctly_worn_dir)):
            if filename.endswith('.mp4'):
                video_path = os.path.join(correctly_worn_dir, filename)
                frames = self.extract_frames(video_path)
                
                if len(frames) > 0:
                    # Pad or truncate to fixed number of frames
                    if len(frames) < self.frames_per_video:
                        # Pad with last frame
                        while len(frames) < self.frames_per_video:
                            frames.append(frames[-1])
                    elif len(frames) > self.frames_per_video:
                        frames = frames[:self.frames_per_video]
                    
                    X.append(np.array(frames))
                    y.append(1)  # Correctly worn
        
        print("Loading incorrectly worn videos...")
        for filename in tqdm(os.listdir(incorrectly_worn_dir)):
            if filename.endswith('.mp4'):
                video_path = os.path.join(incorrectly_worn_dir, filename)
                frames = self.extract_frames(video_path)
                
                if len(frames) > 0:
                    # Pad or truncate to fixed number of frames
                    if len(frames) < self.frames_per_video:
                        # Pad with last frame
                        while len(frames) < self.frames_per_video:
                            frames.append(frames[-1])
                    elif len(frames) > self.frames_per_video:
                        frames = frames[:self.frames_per_video]
                    
                    X.append(np.array(frames))
                    y.append(0)  # Incorrectly worn
        
        return np.array(X), np.array(y)
    
    def create_model(self):
        """
        Create a CNN model for video classification
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Use simple_classifier.py instead.")
            
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(self.frames_per_video, *self.img_size, 3)),
            
            # TimeDistributed CNN layers for frame processing
            layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu')),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu')),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            layers.TimeDistributed(layers.Conv2D(128, (3, 3), activation='relu')),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            
            # Flatten each frame's features
            layers.TimeDistributed(layers.Flatten()),
            
            # LSTM for temporal modeling
            layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5),
            
            # Dense layers
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, X, y, validation_split=0.2, epochs=20, batch_size=4):
        """
        Train the model
        
        Args:
            X: Feature array
            y: Labels array
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Use simple_classifier.py instead.")
            
        if self.model is None:
            self.create_model()
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model and generate classification report
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Predictions
        y_pred_prob = self.model.predict(X_test)
        y_pred = (y_pred_prob > 0.5).astype(int).flatten()
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if PLOTTING_AVAILABLE:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        return y_pred, y_pred_prob
    
    def predict_video(self, video_path):
        """
        Predict whether device is correctly worn in a single video
        
        Args:
            video_path: Path to the video file
            
        Returns:
            prediction: 0 (incorrectly worn) or 1 (correctly worn)
            confidence: Prediction confidence score
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            raise ValueError(f"Could not extract frames from {video_path}")
        
        # Pad or truncate frames
        if len(frames) < self.frames_per_video:
            while len(frames) < self.frames_per_video:
                frames.append(frames[-1])
        elif len(frames) > self.frames_per_video:
            frames = frames[:self.frames_per_video]
        
        # Prepare input
        X = np.array([frames])  # Add batch dimension
        
        # Predict
        prob = self.model.predict(X)[0][0]
        prediction = 1 if prob > 0.5 else 0
        
        return prediction, prob
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
        self.model.save(filepath)
        
    def load_model(self, filepath):
        """Load a trained model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Use simple_classifier.py instead.")
        self.model = keras.models.load_model(filepath)
