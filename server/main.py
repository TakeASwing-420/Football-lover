from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import tempfile
import os
import torch

from model.lofi2lofi_model import Decoder as Lofi2LofiDecoder
from lofi2lofi_generate import decode

device = "cpu"
app = Flask(__name__)
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["30 per minute"])

# Load model once
checkpoint = "checkpoints/lofi2lofi_decoder.pth"
print("Loading lofi model...", end=" ")
model = Lofi2LofiDecoder(device=device)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.to(device)
model.eval()
print(f"Loaded {checkpoint}.")


@app.route('/')
def home():
    return 'Server running'


@app.route('/decode', methods=['POST'])
def decode_endpoint():
    if 'video' not in request.files:
        response = jsonify({'error': 'No video uploaded'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 400

    video_file = request.files['video']
    if video_file.filename == '':
        response = jsonify({'error': 'Empty filename'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        video_path = tmp.name
        video_file.save(video_path)

    try:
        result = decode(model, video_path)
        if result is None:
            response = jsonify({'error': 'Input video is not lofifiable.'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 422
        elif result == "mood_tag not present":
            response = jsonify({'error': 'Mood_tag not found.'})
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response, 422

        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 201
    except Exception as e:
        response = jsonify({'error': f'Server error: {str(e)}'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 500
    finally:
        os.remove(video_path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
