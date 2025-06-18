# Stroke Rehabilitation Device Placement Classifier

A machine learning system to automatically detect whether a stroke rehabilitation device is worn correctly or incorrectly using video analysis.

## Overview

This project uses deep learning to classify videos of patients wearing rehabilitation devices into two categories:
- **Correctly worn**: Device is properly positioned
- **Incorrectly worn**: Device is improperly positioned

The system uses a combination of Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal modeling.

## Dataset

- **Total videos**: 21
- **Correctly worn**: 9 videos
- **Incorrectly worn**: 12 videos
- **Video properties**: Mixed resolutions (848x478, 1280x720), 30-60 fps, 3-4 seconds duration

## Model Architecture

The model consists of:
1. **TimeDistributed CNN layers**: Extract spatial features from individual frames
2. **LSTM layer**: Model temporal relationships between frames
3. **Dense layers**: Final classification with dropout for regularization

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the Model

```bash
python train_model.py
```

This will:
- Load and preprocess the video dataset
- Train the CNN-LSTM model
- Save the trained model as `device_placement_model.h5`
- Generate training history plots

### 2. Predict Single Video

```bash
python predict_video.py "path/to/video.mp4"
```

Example:
```bash
python predict_video.py "correctly worn/16.mp4"
```

### 3. Evaluate Model Performance

```bash
python evaluate_model.py
```

This will:
- Test the model on all videos in the dataset
- Generate detailed accuracy metrics
- Save results to `evaluation_results.csv`
- Show misclassified videos

## Files Description

- `video_classifier.py`: Main classifier class with all ML functionality
- `train_model.py`: Training script
- `predict_video.py`: Single video prediction script
- `evaluate_model.py`: Comprehensive model evaluation
- `analyze_dataset.py`: Dataset analysis utility
- `requirements.txt`: Python dependencies

## Model Features

- **Frame extraction**: Intelligently samples frames from videos
- **Data augmentation**: Built-in preprocessing and normalization
- **Temporal modeling**: LSTM captures movement patterns over time
- **Robust architecture**: Handles variable video lengths and resolutions
- **Early stopping**: Prevents overfitting during training
- **Confidence scoring**: Provides prediction confidence levels

## Performance Metrics

The model provides:
- Overall accuracy
- Per-class accuracy (correctly/incorrectly worn)
- Confidence scores for predictions
- Confusion matrix visualization
- Detailed evaluation reports

## Technical Details

- **Input size**: 224x224 pixels per frame
- **Frames per video**: 16 (evenly sampled)
- **Batch size**: 2-4 (optimized for small dataset)
- **Training epochs**: 25 with early stopping
- **Validation split**: 20%

## Output Files

After training and evaluation:
- `device_placement_model.h5`: Trained model
- `training_history.png`: Training/validation curves
- `confusion_matrix.png`: Classification confusion matrix
- `evaluation_results.csv`: Detailed evaluation results

## Usage Examples

### Quick Prediction
```python
from video_classifier import DevicePlacementClassifier

classifier = DevicePlacementClassifier()
classifier.load_model('device_placement_model.h5')

prediction, confidence = classifier.predict_video('test_video.mp4')
print(f"Prediction: {'Correctly worn' if prediction == 1 else 'Incorrectly worn'}")
print(f"Confidence: {confidence:.3f}")
```

### Custom Training
```python
classifier = DevicePlacementClassifier(img_size=(224, 224), frames_per_video=16)
X, y = classifier.load_dataset('correctly worn', 'incorrectly worn')
history = classifier.train(X, y, epochs=30, batch_size=4)
```

## Requirements

- Python 3.7+
- OpenCV 4.8+
- TensorFlow 2.13+
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn

## Notes

- The model is optimized for the specific dataset characteristics
- Small dataset size may require careful validation
- Consider data augmentation for improved generalization
- GPU acceleration recommended for faster training
