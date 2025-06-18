"""
Manual training - step by step
"""

import os
import pickle
from simple_classifier import SimpleDeviceClassifier

# Step 1: Initialize
print("Step 1: Initializing classifier...")
classifier = SimpleDeviceClassifier(img_size=(128, 128), frames_per_video=4)

# Step 2: Load data
print("Step 2: Loading dataset...")
X, y = classifier.load_dataset('correctly worn', 'incorrectly worn')
print(f"Loaded {len(X)} samples")

# Step 3: Train
print("Step 3: Training model...")
accuracy = classifier.train(X, y, model_type='random_forest', test_size=0.3)
print(f"Training accuracy: {accuracy:.3f}")

# Step 4: Manual save
print("Step 4: Saving model manually...")
model_data = {
    'model': classifier.model,
    'scaler': classifier.scaler,
    'img_size': classifier.img_size,
    'frames_per_video': classifier.frames_per_video
}

with open('device_classifier_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model saved successfully!")

# Step 5: Test
print("Step 5: Testing prediction...")
test_video = 'correctly worn/16.mp4'
if os.path.exists(test_video):
    pred, conf = classifier.predict_video(test_video)
    print(f"Test prediction: {'Correct' if pred == 1 else 'Incorrect'} (confidence: {conf:.3f})")

print("All done!")
