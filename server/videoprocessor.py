import cv2
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import os

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Define feature label sets
valence_labels = ["sad", "neutral", "happy"]
tempo_labels = ["slow tempo", "medium tempo", "fast tempo"]
energy_labels = ["low energy", "moderate energy", "high energy"]
swing_labels = ["mechanical rhythm", "slightly swung rhythm", "very swung rhythm"]
mood_labels = [
    "chill and lofi", "fast-paced and energetic", "dark and tense", "bright and happy", 
    "calm and ambient", "uplifting and hopeful", "mysterious and cinematic", "intense and dramatic",
    "nostalgic and sentimental", "playful and fun", "romantic and emotional", "epic and grand", 
    "quirky and experimental", "spooky and eerie", "adventurous and exploratory", "peaceful and serene", 
    "introspective and thoughtful", "joyful and celebratory", "melancholic and reflective", 
    "energetic and upbeat", "dramatic and powerful", "epic and climatic"
]

# Frame extraction with checks and logging
def extract_frames(video_path, num_frames=10):
    if not os.path.isfile(video_path):
        print(f"[ERROR] File not found: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        print(f"[ERROR] Video has zero frames: {video_path}")
        return []

    interval = max(frame_count // num_frames, 1)
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARNING] Could not read frame {i * interval}")
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(rgb)

    cap.release()
    print(f"[INFO] Extracted {len(frames)} frames from '{video_path}'")
    return frames

# Helper: compute similarities between image embedding and text prompts
def rank_labels(image_embedding, text_labels):
    inputs = processor(text=text_labels, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    similarities = cosine_similarity(image_embedding.cpu(), text_features.cpu())[0]
    return similarities

# Main predictor with safety checks
def predict_music_features(video_path):
    frames = extract_frames(video_path)
    if len(frames) == 0:
        raise ValueError("❌ No frames extracted from video. Please check the file path and format.")

    inputs = processor(images=frames, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)

    image_embedding = image_features.mean(dim=0, keepdim=True)

    # Compute similarity-based features
    valence_scores = rank_labels(image_embedding, valence_labels)
    tempo_scores = rank_labels(image_embedding, tempo_labels)
    energy_scores = rank_labels(image_embedding, energy_labels)
    swing_scores = rank_labels(image_embedding, swing_labels)
    mood_scores = rank_labels(image_embedding, mood_labels)

    # Normalize scores to [0, 1]
    valence = np.dot(valence_scores, [0.0, 0.5, 1.0]) / valence_scores.sum()
    tempo_idx = int(np.argmax(tempo_scores))
    energy = np.dot(energy_scores, [0.0, 0.5, 1.0]) / energy_scores.sum()
    swing = np.dot(swing_scores, [0.0, 0.5, 1.0]) / swing_scores.sum()
    mood_tag = mood_labels[int(np.argmax(mood_scores))]

    tempo_str = tempo_labels[tempo_idx].replace(" tempo", "")

    return {
        "valence": round(valence, 3),
        "tempo": tempo_str,
        "energy": round(energy, 3),
        "swing": round(swing, 3),
        "mood_tag": mood_tag
    }

# Example usage
if __name__ == "__main__":
    video_path = os.path.join(os.path.dirname(__file__), "7232007-uhd_2160_3840_25fps.mp4")
    try:
        features = predict_music_features(video_path)
        print(features)
    except Exception as e:
        print(f"[FATAL] {str(e)}")
