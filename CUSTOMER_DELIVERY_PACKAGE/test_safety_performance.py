"""
Manual Safety Performance Testing Script
Tests how well the model catches dangerous situations (incorrectly worn devices)
"""

import os
from simple_classifier import SimpleDeviceClassifier

def test_safety_performance():
    print("🛡️ SAFETY PERFORMANCE TEST")
    print("=" * 50)
    print("Testing how well the model catches dangerous situations...")
    print()
    
    # Load model
    classifier = SimpleDeviceClassifier()
    classifier.load_model('device_classifier_model.pkl')
    
    # Test all incorrectly worn videos (dangerous situations)
    incorrectly_worn_dir = 'incorrectly worn'
    incorrectly_worn_files = [f for f in os.listdir(incorrectly_worn_dir) if f.endswith('.mp4')]
    
    caught_dangerous = 0
    missed_dangerous = 0
    total_dangerous = len(incorrectly_worn_files)
    
    print(f"Testing {total_dangerous} dangerous situations (incorrectly worn devices):")
    print("-" * 60)
    
    for i, filename in enumerate(incorrectly_worn_files, 1):
        video_path = os.path.join(incorrectly_worn_dir, filename)
        prediction, confidence = classifier.predict_video(video_path)
        
        if prediction == 0:  # Correctly identified as incorrectly worn
            caught_dangerous += 1
            status = "✅ CAUGHT"
            safety_impact = "SAFE"
        else:  # Missed - predicted as correctly worn
            missed_dangerous += 1
            status = "❌ MISSED"
            safety_impact = "⚠️ DANGER"
        
        print(f"{i:2d}. {filename:15s}: {status} (conf: {confidence:.3f}) - {safety_impact}")
    
    # Calculate safety metrics
    safety_detection_rate = (caught_dangerous / total_dangerous) * 100
    false_negative_rate = (missed_dangerous / total_dangerous) * 100
    
    print("\n" + "=" * 60)
    print("🛡️ SAFETY PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Total dangerous situations: {total_dangerous}")
    print(f"Dangerous situations caught: {caught_dangerous}")
    print(f"Dangerous situations missed: {missed_dangerous}")
    print(f"Safety Detection Rate: {safety_detection_rate:.1f}%")
    print(f"False Negative Rate: {false_negative_rate:.1f}%")
    
    # Safety assessment
    print("\n📊 SAFETY ASSESSMENT:")
    if safety_detection_rate >= 90:
        print("🟢 EXCELLENT safety performance - Very safe for patients")
    elif safety_detection_rate >= 80:
        print("🟡 GOOD safety performance - Acceptable for medical use")
    elif safety_detection_rate >= 70:
        print("🟠 FAIR safety performance - Consider improvements")
    else:
        print("🔴 POOR safety performance - Not recommended for medical use")
    
    print(f"\n💡 Clinical Impact:")
    print(f"   - {caught_dangerous} patients would be alerted to fix device placement")
    print(f"   - {missed_dangerous} patients might continue with incorrect placement")
    
    return safety_detection_rate, false_negative_rate

if __name__ == "__main__":
    test_safety_performance()
