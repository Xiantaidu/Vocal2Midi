import pathlib
import textgrid

HTK_PAD_VAL = 10000000

class Exporter:
    def __init__(self, predictions, output_folder=None):
        self.predictions = predictions
        self.output_folder = pathlib.Path(output_folder) if output_folder else None

    def save_textgrids(self):
        print("Saving TextGrids...")
        for wav_path, wav_length, words in self.predictions:
            wav_path = pathlib.Path(wav_path)
            tg = textgrid.TextGrid(minTime=0, maxTime=wav_length)
            word_tier = textgrid.IntervalTier(name="words", minTime=0.0, maxTime=wav_length)
            ph_tier = textgrid.IntervalTier(name="phones", minTime=0.0, maxTime=wav_length)

            for word in words:
                word_tier.add(minTime=word.start, maxTime=word.end, mark=word.text)
                for phoneme in word.phonemes:
                    ph_tier.add(minTime=max(0, phoneme.start), maxTime=phoneme.end, mark=phoneme.text)

            tg.append(word_tier)
            tg.append(ph_tier)

            if self.output_folder is not None:
                tg_path = self.output_folder / "TextGrid" / wav_path.with_suffix(".TextGrid").name
            else:
                tg_path = wav_path.parent / "TextGrid" / wav_path.with_suffix(".TextGrid").name

            tg_path.parent.mkdir(parents=True, exist_ok=True)
            tg.write(tg_path)

    def save_htk(self):
        print("Saving HTK Labels...")
        w_out, ph_out = "", ""
        for wav_path, wav_length, words in self.predictions:
            wav_path = pathlib.Path(wav_path)

            for word in words:
                w_start = int(float(word.start) * HTK_PAD_VAL)
                w_end = int(float(word.end) * HTK_PAD_VAL)
                w_out += f"{w_start} {w_end} {word.text}\n"
                for phoneme in word.phonemes:
                    ph_start = int(float(phoneme.start) * HTK_PAD_VAL)
                    ph_end = int(float(phoneme.end) * HTK_PAD_VAL)
                    ph_out += f"{ph_start} {ph_end} {phoneme.text}\n"

            if self.output_folder is not None:
                htk_ph_path = self.output_folder / "HTK" / "Phones" / wav_path.with_suffix(".lab").name
                htk_w_path = self.output_folder / "HTK" / "Words" / wav_path.with_suffix(".lab").name
            else:
                htk_ph_path = wav_path.parent / "HTK" / "Phones" / wav_path.with_suffix(".lab").name
                htk_w_path = wav_path.parent / "HTK" / "Words" / wav_path.with_suffix(".lab").name

            htk_ph_path.parent.mkdir(parents=True, exist_ok=True)
            htk_w_path.parent.mkdir(parents=True, exist_ok=True)

            with open(htk_ph_path, 'w', encoding='utf-8') as pho:
                pho.write(ph_out)
            with open(htk_w_path, 'w', encoding='utf-8') as wo:
                wo.write(w_out)

    def export(self, out_formats):
        if 'textgrid' in out_formats:
            self.save_textgrids()
        if 'htk' in out_formats:
            self.save_htk()
