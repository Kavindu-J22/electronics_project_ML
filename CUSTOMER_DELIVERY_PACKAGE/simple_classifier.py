"""
Simple Video Classifier using Traditional ML
This version uses feature extraction + traditional ML algorithms
as a fallback when deep learning libraries are not available.
"""

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# Optional imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Plots will be skipped.")

try:
    from tqdm import tqdm
except ImportError:
    # Simple replacement for tqdm
    def tqdm(iterable, desc=None):
        if desc:
            print(f"{desc}...")
        return iterable

class SimpleDeviceClassifier:
    def __init__(self, img_size=(224, 224), frames_per_video=8):
        """
        Initialize the simple classifier
        
        Args:
            img_size: Target image size for frames (width, height)
            frames_per_video: Number of frames to extract from each video
        """
        self.img_size = img_size
        self.frames_per_video = frames_per_video
        self.model = None
        self.scaler = StandardScaler()
        self.class_names = ['incorrectly_worn', 'correctly_worn']
        
    def extract_features_from_frame(self, frame):
        """
        Extract features from a single frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Feature vector
        """
        features = []
        
        # Convert to grayscale for some features
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # 1. Color histogram features (RGB)
        for i in range(3):  # R, G, B channels
            hist = cv2.calcHist([frame], [i], None, [32], [0, 256])
            features.extend(hist.flatten())
        
        # 2. Grayscale histogram
        gray_hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        features.extend(gray_hist.flatten())
        
        # 3. Edge features using Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        
        # 4. Texture features using Local Binary Pattern approximation
        # Simple texture measure: standard deviation of pixel intensities
        texture_std = np.std(gray)
        features.append(texture_std)
        
        # 5. Brightness and contrast
        brightness = np.mean(gray)
        contrast = np.std(gray)
        features.extend([brightness, contrast])
        
        # 6. Shape features - contour analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Number of contours
            num_contours = len(contours)
            # Largest contour area
            largest_area = max([cv2.contourArea(c) for c in contours])
            # Average contour area
            avg_area = np.mean([cv2.contourArea(c) for c in contours])
        else:
            num_contours = 0
            largest_area = 0
            avg_area = 0
        
        features.extend([num_contours, largest_area, avg_area])
        
        # 7. Spatial moments
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            cx = moments['m10'] / moments['m00']  # Centroid x
            cy = moments['m01'] / moments['m00']  # Centroid y
        else:
            cx = cy = 0
        features.extend([cx, cy])
        
        return np.array(features)
    
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
                frames.append(frame)
        
        cap.release()
        return frames
    
    def extract_video_features(self, video_path):
        """
        Extract features from entire video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Feature vector for the video
        """
        frames = self.extract_frames(video_path)
        
        if len(frames) == 0:
            return None
        
        video_features = []
        
        # Extract features from each frame
        frame_features = []
        for frame in frames:
            features = self.extract_features_from_frame(frame)
            frame_features.append(features)
        
        frame_features = np.array(frame_features)
        
        # Aggregate features across frames
        # Statistical measures across time
        video_features.extend(np.mean(frame_features, axis=0))  # Mean
        video_features.extend(np.std(frame_features, axis=0))   # Standard deviation
        video_features.extend(np.max(frame_features, axis=0))   # Maximum
        video_features.extend(np.min(frame_features, axis=0))   # Minimum
        
        # Temporal features - differences between consecutive frames
        if len(frame_features) > 1:
            frame_diffs = np.diff(frame_features, axis=0)
            video_features.extend(np.mean(frame_diffs, axis=0))  # Mean change
            video_features.extend(np.std(frame_diffs, axis=0))   # Variability of change
        else:
            # If only one frame, pad with zeros
            video_features.extend(np.zeros(frame_features.shape[1]))
            video_features.extend(np.zeros(frame_features.shape[1]))
        
        return np.array(video_features)
    
    def load_dataset(self, correctly_worn_dir, incorrectly_worn_dir):
        """
        Load and preprocess the video dataset
        
        Args:
            correctly_worn_dir: Directory containing correctly worn videos
            incorrectly_worn_dir: Directory containing incorrectly worn videos
            
        Returns:
            X: Feature array
            y: Labels array
        """
        X = []
        y = []
        
        print("Loading correctly worn videos...")
        for filename in tqdm(os.listdir(correctly_worn_dir)):
            if filename.endswith('.mp4'):
                video_path = os.path.join(correctly_worn_dir, filename)
                features = self.extract_video_features(video_path)
                
                if features is not None:
                    X.append(features)
                    y.append(1)  # Correctly worn
        
        print("Loading incorrectly worn videos...")
        for filename in tqdm(os.listdir(incorrectly_worn_dir)):
            if filename.endswith('.mp4'):
                video_path = os.path.join(incorrectly_worn_dir, filename)
                features = self.extract_video_features(video_path)
                
                if features is not None:
                    X.append(features)
                    y.append(0)  # Incorrectly worn
        
        return np.array(X), np.array(y)
    
    def train(self, X, y, model_type='random_forest', test_size=0.2):
        """
        Train the model
        
        Args:
            X: Feature array
            y: Labels array
            model_type: Type of model ('random_forest', 'svm')
            test_size: Fraction of data to use for testing
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Feature dimension: {X.shape[1]}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        print(f"Training {model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nTraining completed!")
        print(f"Test Accuracy: {accuracy:.3f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Incorrect  Correct")
        print(f"Actual Incorrect    {cm[0,0]:3d}      {cm[0,1]:3d}")
        print(f"       Correct      {cm[1,0]:3d}      {cm[1,1]:3d}")

        if PLOTTING_AVAILABLE:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('simple_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("Plotting not available - confusion matrix displayed as text above.")
        
        return accuracy
    
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
        
        features = self.extract_video_features(video_path)
        
        if features is None:
            raise ValueError(f"Could not extract features from {video_path}")
        
        # Scale features
        features_scaled = self.scaler.transform([features])
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            confidence = proba[1] if prediction == 1 else proba[0]
        else:
            confidence = 0.5  # Default confidence for models without probability
        
        return prediction, confidence
    
    def save_model(self, filepath):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'img_size': self.img_size,
            'frames_per_video': self.frames_per_video
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and scaler"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.img_size = model_data['img_size']
        self.frames_per_video = model_data['frames_per_video']
        
        print(f"Model loaded from {filepath}")
