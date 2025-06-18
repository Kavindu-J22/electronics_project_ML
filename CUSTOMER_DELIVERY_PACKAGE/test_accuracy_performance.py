"""
Manual Accuracy Performance Testing Script
Tests overall accuracy, precision, recall, and confidence reliability
"""

import os
from simple_classifier import SimpleDeviceClassifier

def test_accuracy_performance():
    print("🎯 ACCURACY PERFORMANCE TEST")
    print("=" * 50)
    print("Testing overall accuracy and classification metrics...")
    print()
    
    # Load model
    classifier = SimpleDeviceClassifier()
    classifier.load_model('device_classifier_model.pkl')
    
    # Test all videos
    results = []
    
    # Test correctly worn videos
    print("Testing CORRECTLY WORN videos:")
    print("-" * 40)
    correctly_worn_files = [f for f in os.listdir('correctly worn') if f.endswith('.mp4')]
    
    for i, filename in enumerate(correctly_worn_files, 1):
        video_path = f'correctly worn/{filename}'
        prediction, confidence = classifier.predict_video(video_path)
        
        correct = (prediction == 1)  # Should predict correctly worn (1)
        status = "✅ CORRECT" if correct else "❌ WRONG"
        
        results.append({
            'filename': filename,
            'true_label': 1,  # correctly worn
            'predicted': prediction,
            'confidence': confidence,
            'correct': correct
        })
        
        print(f"{i:2d}. {filename:15s}: {status} (conf: {confidence:.3f})")
    
    print(f"\nTesting INCORRECTLY WORN videos:")
    print("-" * 40)
    incorrectly_worn_files = [f for f in os.listdir('incorrectly worn') if f.endswith('.mp4')]
    
    for i, filename in enumerate(incorrectly_worn_files, 1):
        video_path = f'incorrectly worn/{filename}'
        prediction, confidence = classifier.predict_video(video_path)
        
        correct = (prediction == 0)  # Should predict incorrectly worn (0)
        status = "✅ CORRECT" if correct else "❌ WRONG"
        
        results.append({
            'filename': filename,
            'true_label': 0,  # incorrectly worn
            'predicted': prediction,
            'confidence': confidence,
            'correct': correct
        })
        
        print(f"{i:2d}. {filename:15s}: {status} (conf: {confidence:.3f})")
    
    # Calculate accuracy metrics
    total_videos = len(results)
    correct_predictions = sum(1 for r in results if r['correct'])
    overall_accuracy = correct_predictions / total_videos
    
    # Per-class accuracy
    correctly_worn_results = [r for r in results if r['true_label'] == 1]
    incorrectly_worn_results = [r for r in results if r['true_label'] == 0]
    
    correctly_worn_accuracy = sum(1 for r in correctly_worn_results if r['correct']) / len(correctly_worn_results)
    incorrectly_worn_accuracy = sum(1 for r in incorrectly_worn_results if r['correct']) / len(incorrectly_worn_results)
    
    # Confusion matrix components
    tp = sum(1 for r in results if r['true_label'] == 1 and r['predicted'] == 1)  # True positive
    tn = sum(1 for r in results if r['true_label'] == 0 and r['predicted'] == 0)  # True negative
    fp = sum(1 for r in results if r['true_label'] == 0 and r['predicted'] == 1)  # False positive
    fn = sum(1 for r in results if r['true_label'] == 1 and r['predicted'] == 0)  # False negative
    
    # Calculate precision, recall, F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Confidence analysis
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    high_conf_results = [r for r in results if r['confidence'] > 0.8]
    high_conf_accuracy = sum(1 for r in high_conf_results if r['correct']) / len(high_conf_results) if high_conf_results else 0
    
    print("\n" + "=" * 60)
    print("🎯 ACCURACY PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Total videos tested: {total_videos}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Overall accuracy: {overall_accuracy:.3f} ({overall_accuracy*100:.1f}%)")
    print()
    print("Per-Class Performance:")
    print(f"  Correctly worn accuracy: {correctly_worn_accuracy:.3f} ({correctly_worn_accuracy*100:.1f}%)")
    print(f"  Incorrectly worn accuracy: {incorrectly_worn_accuracy:.3f} ({incorrectly_worn_accuracy*100:.1f}%)")
    print()
    print("Detailed Metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1_score:.3f}")
    print()
    print("Confidence Analysis:")
    print(f"  Average confidence: {avg_confidence:.3f}")
    print(f"  High confidence predictions (>0.8): {len(high_conf_results)}/{total_videos}")
    print(f"  High confidence accuracy: {high_conf_accuracy:.3f} ({high_conf_accuracy*100:.1f}%)")
    
    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                 Incorrect  Correct")
    print(f"Actual Incorrect    {tn:3d}      {fp:3d}")
    print(f"       Correct      {fn:3d}      {tp:3d}")
    
    # Accuracy assessment
    print("\n📊 ACCURACY ASSESSMENT:")
    if overall_accuracy >= 0.9:
        print("🟢 EXCELLENT accuracy - Outstanding performance")
    elif overall_accuracy >= 0.8:
        print("🟡 GOOD accuracy - Strong performance")
    elif overall_accuracy >= 0.7:
        print("🟠 FAIR accuracy - Acceptable performance")
    else:
        print("🔴 POOR accuracy - Needs improvement")
    
    print(f"\n💡 Clinical Interpretation:")
    print(f"   - {correctly_worn_accuracy*100:.1f}% of correctly worn devices identified")
    print(f"   - {incorrectly_worn_accuracy*100:.1f}% of incorrectly worn devices caught")
    print(f"   - {high_conf_accuracy*100:.1f}% accuracy when model is confident")
    
    # Show misclassified videos
    misclassified = [r for r in results if not r['correct']]
    if misclassified:
        print(f"\n⚠️ Misclassified Videos ({len(misclassified)}):")
        for r in misclassified:
            true_label = "correctly worn" if r['true_label'] == 1 else "incorrectly worn"
            pred_label = "correctly worn" if r['predicted'] == 1 else "incorrectly worn"
            print(f"   {r['filename']}: True={true_label}, Predicted={pred_label} (conf: {r['confidence']:.3f})")
    
    return overall_accuracy, precision, recall, f1_score

if __name__ == "__main__":
    test_accuracy_performance()
