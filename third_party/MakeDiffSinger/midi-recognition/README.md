# MIDI Recognition

## 1. merge_wavs.py

Merge short audio clips into long audio segments of similar length (e.g. 4 min) and a fixed sampling rate (e.g. 16000) and save the timestamps into tags.json.

## 2. extract_midi.py

Extract MIDI sequences from of OpenSVIP json files, split them back into short clips according to tags.json, and add them into transcriptions.csv.

