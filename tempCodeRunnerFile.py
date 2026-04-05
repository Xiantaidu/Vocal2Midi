        kwargs = {
            'audio_files': audio_files,
            'game_model_path_str': self.global_settings.game_model_edit.text(),
            'device': self.device_combo.currentText(),
            'hfa_model_path_str': self.global_settings.hfa_model_edit.text(),
            'asr_model_path_str': self.global_settings.asr_model_edit.text(),
            'language': self.lang_combo.currentText(),
            'original_lyrics': self.lyrics_edit.toPlainText().strip(),
            'output_formats': output_formats,
            'slicing_method': self.slicing_combo.currentText(),
            'tempo': self.tempo_spin.value(),
            'quantization_step': _parse_quantization(self.quantize_combo.currentText()),
            'pitch_format': self.pitch_combo.currentText(),
            'round_pitch': self.cb_round.isChecked(),
            'seg_threshold': self.global_settings.seg_thresh_spin.value(),
            'seg_radius': self.global_settings.seg_rad_spin.value(),
            't0': self.global_settings.t0_spin.value(),
            'nsteps': self.global_settings.nsteps_spin.value(),
            'est_threshold': self.global_settings.est_thresh_spin.value(),
            'batch_size': self.global_settings.batch_spin.value(),
            'asr_batch_size': self.global_settings.asr_batch_spin.value()
        }
        