import jsonpickle

from model.constants import *


class Output:
    def __init__(self, title, pred_chords, pred_notes, pred_tempo, pred_key, pred_mode, pred_valence, pred_energy, pred_swing):
        chords = pred_chords.argmax(dim=2)[0].tolist()
        notes = pred_notes.argmax(dim=2)[0].cpu().numpy()

        chords.append(CHORD_END_TOKEN)
        cut_off_point = chords.index(CHORD_END_TOKEN)
        chords = chords[:cut_off_point]  # cut off end token
        notes = notes[:cut_off_point * NOTES_PER_CHORD]
        melodies = notes.reshape(-1, NOTES_PER_CHORD)
        bpm = {
             "slow": 70,
             "medium": 85,
            "fast": 100
        }[pred_tempo]
        energy = min(1, max(0, pred_energy.item()))
        valence = min(1, max(0, pred_valence.item()))
        swing = min(1, max(0, pred_swing.item()))
        self.title = title
        self.key = pred_key.argmax().item() + 1
        self.mode = pred_mode.argmax().item() + 1
        self.bpm = round(bpm)
        self.energy = energy
        self.valence = valence
        self.chords = chords
        self.melodies = [x.tolist() for x in [*melodies]]
        self.swing = swing

    def to_json(self):
        json = jsonpickle.encode(self, unpicklable=False)
        return json
