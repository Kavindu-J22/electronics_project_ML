"""
Comprehensive evaluation script for the simple model
"""

import os
import numpy as np
import pandas as pd
from simple_classifier import SimpleDeviceClassifier

def evaluate_all_videos():
    """Evaluate model on all videos in the dataset"""
    
    model_path = 'device_classifier_model.pkl'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first.")
        return
    
    # Initialize classifier and load model
    print("Loading trained model...")
    classifier = SimpleDeviceClassifier()
    classifier.load_model(model_path)
    
    # Data directories
    correctly_worn_dir = 'correctly worn'
    incorrectly_worn_dir = 'incorrectly worn'
    
    results = []
    
    print("\nEvaluating correctly worn videos...")
    correctly_worn_files = [f for f in os.listdir(correctly_worn_dir) if f.endswith('.mp4')]
    for i, filename in enumerate(correctly_worn_files, 1):
        video_path = os.path.join(correctly_worn_dir, filename)
        try:
            prediction, confidence = classifier.predict_video(video_path)
            correct = prediction == 1
            results.append({
                'filename': filename,
                'true_label': 'correctly_worn',
                'predicted_label': 'correctly_worn' if prediction == 1 else 'incorrectly_worn',
                'confidence': confidence,
                'correct': correct
            })
            status = "CORRECT" if correct else "WRONG"
            print(f"  {i:2d}. {filename:15s}: {status} (conf: {confidence:.3f})")
        except Exception as e:
            print(f"  {i:2d}. {filename:15s}: ERROR - {str(e)}")
    
    print("\nEvaluating incorrectly worn videos...")
    incorrectly_worn_files = [f for f in os.listdir(incorrectly_worn_dir) if f.endswith('.mp4')]
    for i, filename in enumerate(incorrectly_worn_files, 1):
        video_path = os.path.join(incorrectly_worn_dir, filename)
        try:
            prediction, confidence = classifier.predict_video(video_path)
            correct = prediction == 0
            results.append({
                'filename': filename,
                'true_label': 'incorrectly_worn',
                'predicted_label': 'correctly_worn' if prediction == 1 else 'incorrectly_worn',
                'confidence': confidence,
                'correct': correct
            })
            status = "✅ Correct" if correct else "❌ Wrong"
            print(f"  {i:2d}. {filename:15s}: {status} (conf: {confidence:.3f})")
        except Exception as e:
            print(f"  {i:2d}. {filename:15s}: ❌ Error - {str(e)}")
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Calculate metrics
    total_videos = len(df)
    correct_predictions = df['correct'].sum()
    accuracy = correct_predictions / total_videos
    
    # Detailed analysis
    print("\n" + "="*60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("="*60)
    print(f"Total videos evaluated: {total_videos}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Per-class accuracy
    correctly_worn_df = df[df['true_label'] == 'correctly_worn']
    incorrectly_worn_df = df[df['true_label'] == 'incorrectly_worn']
    
    if len(correctly_worn_df) > 0:
        correctly_worn_acc = correctly_worn_df['correct'].mean()
        print(f"Correctly worn accuracy: {correctly_worn_acc:.3f} ({correctly_worn_acc*100:.1f}%)")
    
    if len(incorrectly_worn_df) > 0:
        incorrectly_worn_acc = incorrectly_worn_df['correct'].mean()
        print(f"Incorrectly worn accuracy: {incorrectly_worn_acc:.3f} ({incorrectly_worn_acc*100:.1f}%)")
    
    # Confidence analysis
    avg_confidence = df['confidence'].mean()
    print(f"Average confidence: {avg_confidence:.3f}")
    
    # High confidence predictions
    high_conf_df = df[df['confidence'] > 0.8]
    if len(high_conf_df) > 0:
        high_conf_acc = high_conf_df['correct'].mean()
        print(f"High confidence (>0.8) predictions: {len(high_conf_df)}/{total_videos}")
        print(f"High confidence accuracy: {high_conf_acc:.3f} ({high_conf_acc*100:.1f}%)")
    
    # Medium confidence predictions
    med_conf_df = df[(df['confidence'] > 0.6) & (df['confidence'] <= 0.8)]
    if len(med_conf_df) > 0:
        med_conf_acc = med_conf_df['correct'].mean()
        print(f"Medium confidence (0.6-0.8) predictions: {len(med_conf_df)}/{total_videos}")
        print(f"Medium confidence accuracy: {med_conf_acc:.3f} ({med_conf_acc*100:.1f}%)")
    
    # Low confidence predictions
    low_conf_df = df[df['confidence'] <= 0.6]
    if len(low_conf_df) > 0:
        low_conf_acc = low_conf_df['correct'].mean()
        print(f"Low confidence (≤0.6) predictions: {len(low_conf_df)}/{total_videos}")
        print(f"Low confidence accuracy: {low_conf_acc:.3f} ({low_conf_acc*100:.1f}%)")
    
    # Confusion matrix
    tp = len(df[(df['true_label'] == 'correctly_worn') & (df['predicted_label'] == 'correctly_worn')])
    tn = len(df[(df['true_label'] == 'incorrectly_worn') & (df['predicted_label'] == 'incorrectly_worn')])
    fp = len(df[(df['true_label'] == 'incorrectly_worn') & (df['predicted_label'] == 'correctly_worn')])
    fn = len(df[(df['true_label'] == 'correctly_worn') & (df['predicted_label'] == 'incorrectly_worn')])
    
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Incorrect  Correct")
    print(f"Actual Incorrect    {tn:3d}      {fp:3d}")
    print(f"       Correct      {fn:3d}      {tp:3d}")
    
    # Calculate precision, recall, F1-score
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0
    
    print(f"\nMetrics for 'Correctly Worn' class:")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-Score: {f1_score:.3f}")
    
    # Save detailed results
    try:
        df.to_csv('evaluation_results.csv', index=False)
        print(f"\nDetailed results saved to: evaluation_results.csv")
    except Exception as e:
        print(f"\nCould not save CSV file: {e}")
    
    # Show misclassified videos
    misclassified = df[~df['correct']]
    if len(misclassified) > 0:
        print("\n" + "="*40)
        print("MISCLASSIFIED VIDEOS")
        print("="*40)
        for _, row in misclassified.iterrows():
            print(f"❌ {row['filename']}")
            print(f"   True: {row['true_label']}")
            print(f"   Predicted: {row['predicted_label']}")
            print(f"   Confidence: {row['confidence']:.3f}")
            print()
    else:
        print("\nNo misclassified videos! Perfect accuracy!")
    
    print("="*60)
    
    # Performance summary
    if accuracy >= 0.9:
        print("EXCELLENT performance! Model is working very well.")
    elif accuracy >= 0.8:
        print("GOOD performance! Model is working well.")
    elif accuracy >= 0.7:
        print("FAIR performance! Model shows promise but could be improved.")
    else:
        print("POOR performance! Model needs significant improvement.")
    
    print("\nRecommendations:")
    if accuracy < 0.8:
        print("- Consider collecting more training data")
        print("- Try different feature extraction methods")
        print("- Experiment with different model parameters")
    if len(low_conf_df) > total_videos * 0.3:
        print("- Many low-confidence predictions suggest model uncertainty")
        print("- Consider improving feature quality or model complexity")
    
    return accuracy, df

def main():
    evaluate_all_videos()

if __name__ == "__main__":
    main()
