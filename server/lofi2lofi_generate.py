import os
import torch
from output import *
from videoprocessor import predict_music_features
from typing import Optional
from model.lofi2lofi_model import Lofi2LofiModel

checkers = ["chill and lofi", "bright and happy", "calm and ambient", "uplifting and hopeful", "nostalgic and sentimental", "playful and fun", "romantic and emotional", "peaceful and serene", "melancholic and reflective", "energetic and upbeat", "adventurous and exploratory"]

def decode(model: Lofi2LofiModel) -> Optional[str]:
    mu = torch.randn(1, HIDDEN_SIZE)
    file_path = os.path.join(os.path.dirname(__file__), "6800643-hd_1080_1920_30fps.mp4")

    lofify = predict_music_features(file_path)
    is_lofifiable = any(lofify["mood_tag"]==x for x in checkers)
    
    if is_lofifiable:
        hash, (pred_chords, pred_notes, _, pred_key, pred_mode, _, _) = model.decode(mu)
        output = Output(hash, pred_chords, pred_notes, lofify["tempo"], pred_key, pred_mode, lofify["valence"], lofify["energy"],lofify["swing"])
        json = output.to_json()
        return json
    elif lofify.get("mood_tag"):
        return None
    else:
        return "mood_tag not present"

