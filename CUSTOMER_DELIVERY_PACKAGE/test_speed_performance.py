"""
Manual Speed Performance Testing Script
Tests processing speed and real-time capability
"""

import time
import os
from simple_classifier import SimpleDeviceClassifier

def test_speed_performance():
    print("⚡ SPEED PERFORMANCE TEST")
    print("=" * 50)
    print("Testing processing speed and real-time capability...")
    print()
    
    # Load model
    print("Loading model...")
    start_load = time.time()
    classifier = SimpleDeviceClassifier()
    classifier.load_model('device_classifier_model.pkl')
    load_time = time.time() - start_load
    print(f"Model loading time: {load_time:.3f} seconds")
    print()
    
    # Test videos from both categories
    test_videos = []
    
    # Add correctly worn videos
    correctly_worn_files = [f for f in os.listdir('correctly worn') if f.endswith('.mp4')][:4]
    for f in correctly_worn_files:
        test_videos.append(('correctly worn/' + f, 'correctly_worn'))
    
    # Add incorrectly worn videos  
    incorrectly_worn_files = [f for f in os.listdir('incorrectly worn') if f.endswith('.mp4')][:4]
    for f in incorrectly_worn_files:
        test_videos.append(('incorrectly worn/' + f, 'incorrectly_worn'))
    
    print(f"Testing processing speed on {len(test_videos)} videos:")
    print("-" * 70)
    
    processing_times = []
    total_start = time.time()
    
    for i, (video_path, true_label) in enumerate(test_videos, 1):
        # Time individual prediction
        start_time = time.time()
        prediction, confidence = classifier.predict_video(video_path)
        end_time = time.time()
        
        processing_time = end_time - start_time
        processing_times.append(processing_time)
        
        result = 'CORRECTLY WORN' if prediction == 1 else 'INCORRECTLY WORN'
        filename = video_path.split('/')[-1]
        
        print(f"{i:2d}. {filename:15s}: {result:15s} ({confidence:.3f}) - {processing_time:.3f}s")
    
    total_time = time.time() - total_start
    
    # Calculate speed metrics
    avg_time = sum(processing_times) / len(processing_times)
    min_time = min(processing_times)
    max_time = max(processing_times)
    
    print("\n" + "=" * 60)
    print("⚡ SPEED PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Total videos processed: {len(test_videos)}")
    print(f"Total processing time: {total_time:.3f} seconds")
    print(f"Average time per video: {avg_time:.3f} seconds")
    print(f"Fastest processing: {min_time:.3f} seconds")
    print(f"Slowest processing: {max_time:.3f} seconds")
    print(f"Videos per second: {len(test_videos)/total_time:.1f}")
    
    # Real-time capability assessment
    print("\n📊 SPEED ASSESSMENT:")
    if avg_time < 0.5:
        print("🟢 EXCELLENT speed - Perfect for real-time monitoring")
    elif avg_time < 1.0:
        print("🟡 GOOD speed - Suitable for near real-time use")
    elif avg_time < 2.0:
        print("🟠 FAIR speed - Acceptable for batch processing")
    else:
        print("🔴 SLOW speed - May need optimization")
    
    print(f"\n💡 Real-time Capability:")
    if avg_time < 1.0:
        print(f"   ✅ Can process videos in real-time")
        print(f"   ✅ Suitable for live patient monitoring")
    else:
        print(f"   ⚠️ May have delays in real-time scenarios")
    
    print(f"\n🏥 Clinical Usage:")
    videos_per_minute = 60 / avg_time
    print(f"   - Can analyze ~{videos_per_minute:.0f} videos per minute")
    print(f"   - Processing delay: {avg_time:.3f} seconds per check")
    
    return avg_time, min_time, max_time

if __name__ == "__main__":
    test_speed_performance()
