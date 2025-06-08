import json

import torch
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from model.lofi2lofi_model import Decoder as Lofi2LofiDecoder
from lofi2lofi_generate import decode

device = "cpu"
app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["30 per minute"]
)

lofi2lofi_checkpoint = "checkpoints/lofi2lofi_decoder.pth"
print("Loading lofi model...", end=" ")
lofi2lofi_model = Lofi2LofiDecoder(device=device)
lofi2lofi_model.load_state_dict(torch.load(lofi2lofi_checkpoint, map_location=device))
print(f"Loaded {lofi2lofi_checkpoint}.")
lofi2lofi_model.to(device)
lofi2lofi_model.eval()

@app.route('/')
def home():
    return 'Server running'


@app.route('/decode', methods=['GET'])
def decode_endpoint():
    input = request.args.get('input')
    number_list = json.loads(input)
    result = decode(lofi2lofi_model)
    if result is None:
        return jsonify({'error': 'Input video is not lofifiable.'}), 400
    elif result == "mood_tag not present":
        return jsonify({'error': 'Mood_tag not found.'}), 400
    
    response = jsonify(result)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response, 201

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
