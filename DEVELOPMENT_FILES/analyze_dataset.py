import cv2
import os

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    return {
        'fps': fps,
        'frame_count': frame_count,
        'width': width,
        'height': height,
        'duration': duration
    }

# Analyze videos
correctly_worn_dir = 'correctly worn'
incorrectly_worn_dir = 'incorrectly worn'

print('=== CORRECTLY WORN VIDEOS ===')
for file in os.listdir(correctly_worn_dir)[:3]:
    if file.endswith('.mp4'):
        path = os.path.join(correctly_worn_dir, file)
        info = analyze_video(path)
        if info:
            print(f'{file}: {info["width"]}x{info["height"]}, {info["fps"]:.1f}fps, {info["frame_count"]} frames, {info["duration"]:.1f}s')

print('\n=== INCORRECTLY WORN VIDEOS ===')
for file in os.listdir(incorrectly_worn_dir)[:3]:
    if file.endswith('.mp4'):
        path = os.path.join(incorrectly_worn_dir, file)
        info = analyze_video(path)
        if info:
            print(f'{file}: {info["width"]}x{info["height"]}, {info["fps"]:.1f}fps, {info["frame_count"]} frames, {info["duration"]:.1f}s')

# Count total files
correctly_worn_count = len([f for f in os.listdir(correctly_worn_dir) if f.endswith('.mp4')])
incorrectly_worn_count = len([f for f in os.listdir(incorrectly_worn_dir) if f.endswith('.mp4')])

print(f'\nDataset Summary:')
print(f'Correctly worn videos: {correctly_worn_count}')
print(f'Incorrectly worn videos: {incorrectly_worn_count}')
print(f'Total videos: {correctly_worn_count + incorrectly_worn_count}')
