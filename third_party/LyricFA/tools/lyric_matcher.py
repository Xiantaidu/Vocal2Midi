import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .language_processors import ProcessorFactory, LyricData
from .sequence_aligner import SequenceAligner, calculate_difference_count, SmartHighlighter


@dataclass
class ProcessResult:
    lab_name: str
    matched_text: str
    matched_phonetic: str
    asr_phonetic: List[str]
    asr_text: List[str]
    reason: str


class LyricMatcher:
    def __init__(self, language: str) -> None:
        self.language = language.lower()
        self.processor = ProcessorFactory.create_processor(language)
        self.aligner = SequenceAligner()  # 合并后的对齐器
        self.highlighter = SmartHighlighter(self.aligner)  # 共享同一实例

    def process_lyric_file(self, lyric_path: str) -> LyricData:
        try:
            with open(lyric_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()
        except Exception as error:
            raise IOError(f"Cannot read lyric file {lyric_path}: {str(error)}")

        cleaned_text = self.processor.clean_text(raw_text)
        text_list = self.processor.split_text(cleaned_text)
        phonetic_list = self.processor.get_phonetic_list(text_list)
        return LyricData(text_list, phonetic_list, cleaned_text)

    def process_asr_content(self, lab_content: str) -> Tuple[List[str], List[str]]:
        cleaned_content = self.processor.clean_text(lab_content)
        text_list = self.processor.split_text(cleaned_content)
        phonetic_list = self.processor.get_phonetic_list(text_list)
        return text_list, phonetic_list

    def align_lyric_with_asr(
            self,
            asr_phonetic: List[str],
            lyric_text: List[str],
            lyric_phonetic: List[str]
    ) -> Tuple[str, str, str]:
        matched_text, matched_phonetic, _, _, reason = self.aligner.find_best_match_and_return_lyrics(
            input_pronunciation=asr_phonetic,
            reference_text=lyric_text,
            reference_pronunciation=lyric_phonetic
        )
        return matched_text, matched_phonetic, reason

    @staticmethod
    def save_to_json(json_path: str, text: str, phonetic: str) -> None:
        data = {
            "raw_text": text,
            "lab": phonetic,
            "lab_without_tone": phonetic
        }
        try:
            with open(json_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=3)
        except Exception as error:
            raise IOError(f"Cannot write JSON file {json_path}: {str(error)}")


class LyricMatchingPipeline:
    LYRIC_EXTENSION: str = ".txt"
    LAB_EXTENSION: str = ".lab"
    JSON_EXTENSION: str = ".json"

    def __init__(
            self,
            lyric_folder: str,
            lab_folder: str,
            json_folder: str,
            language: str,
            diff_threshold: int = 5
    ) -> None:
        self.lyric_folder = lyric_folder
        self.lab_folder = lab_folder
        self.json_folder = json_folder
        self.language = language
        self.diff_threshold = diff_threshold
        self.matcher = LyricMatcher(language)

        self.total_files: int = 0
        self.success_count: int = 0
        self.diff_count: int = 0
        self.no_match_count: int = 0
        self.missing_lyrics: List[str] = []

    def add_missing_lyric(self, lyric_name: str) -> None:
        if lyric_name not in self.missing_lyrics:
            self.missing_lyrics.append(lyric_name)

    def print_summary(self) -> None:
        if self.missing_lyrics:
            print(f'Files with missing lyrics: {self.missing_lyrics}')
        print(f'{self.diff_count} files exceed difference threshold ({self.diff_threshold}).')
        print(f'Files with no match: {self.no_match_count}')
        print(f'Total files: {self.total_files}, successfully processed: {self.success_count}.')

    def load_all_lyrics(self) -> Dict[str, LyricData]:
        lyric_dict: Dict[str, LyricData] = {}
        lyric_pattern = f'{self.lyric_folder}/*{self.LYRIC_EXTENSION}'
        for lyric_path in glob.glob(lyric_pattern):
            lyric_name = self._extract_filename_without_extension(lyric_path)
            try:
                lyric_data = self.matcher.process_lyric_file(lyric_path)
                if lyric_data:
                    lyric_dict[lyric_name] = lyric_data
            except Exception as error:
                print(f"Error processing lyric file {lyric_name}: {str(error)}")
        print()
        return lyric_dict

    @staticmethod
    def _extract_filename_without_extension(file_path: str) -> str:
        return os.path.splitext(os.path.basename(file_path))[0]

    def process_single_file(
            self,
            lab_path: str,
            lyric_dict: Dict[str, LyricData]
    ) -> Optional[ProcessResult]:
        lab_name = self._extract_filename_without_extension(lab_path)
        lyric_name = lab_name.rsplit("_", 1)[0]

        if lyric_name not in lyric_dict:
            self.add_missing_lyric(lyric_name)
            print(f"Lab file: {lab_path}\nMissing lyric file: {lyric_name}")
            return None

        try:
            with open(lab_path, 'r', encoding='utf-8') as file:
                lab_content = file.read().strip()
        except Exception as error:
            print(f"Error reading lab file {lab_name}: {str(error)}")
            return None

        lyric_data = lyric_dict[lyric_name]
        asr_text, asr_phonetic = self.matcher.process_asr_content(lab_content)

        if not asr_phonetic:
            print(f"Warning: ASR result empty {lab_name}")
            return None

        matched_text, matched_phonetic, reason = self.matcher.align_lyric_with_asr(
            asr_phonetic=asr_phonetic,
            lyric_text=lyric_data.text_list,
            lyric_phonetic=lyric_data.phonetic_list
        )

        return ProcessResult(
            lab_name=lab_name,
            matched_text=matched_text,
            matched_phonetic=matched_phonetic,
            asr_phonetic=asr_phonetic,
            asr_text=asr_text,
            reason=reason
        )

    def compare_and_save_result(self, result: ProcessResult) -> None:
        if not result.matched_text and not result.matched_phonetic:
            self._handle_no_match(result)
            return

        if self.language == 'zh':
            source_sequence = result.asr_phonetic
            target_sequence = result.matched_phonetic.split()
        else:
            source_sequence = result.asr_text
            target_sequence = result.matched_text.split()

        diff_count = calculate_difference_count(source_sequence, target_sequence)

        if diff_count > self.diff_threshold:
            self._display_differences(result.lab_name, result.matched_text,
                                      result.matched_phonetic, result.asr_phonetic)
            self.diff_count += 1

        json_path = f'{self.json_folder}/{result.lab_name}{self.JSON_EXTENSION}'
        self.matcher.save_to_json(json_path, result.matched_text, result.matched_phonetic)
        self.success_count += 1

    def _handle_no_match(self, result: ProcessResult) -> None:
        self.no_match_count += 1
        self._display_no_match(result.lab_name, result.asr_phonetic, result.reason)
        json_path = f'{self.json_folder}/{result.lab_name}{self.JSON_EXTENSION}'
        self.matcher.save_to_json(json_path, "", "")
        self.success_count += 1

    def _display_differences(
            self,
            lab_name: str,
            matched_text: str,
            matched_phonetic: str,
            asr_phonetic: List[str],
    ) -> None:
        asr_result_str = " ".join(asr_phonetic)
        highlighted_asr, highlighted_phonetic, highlighted_text, operation_count = (
            self.matcher.highlighter.highlight_differences(
                asr_result_str, matched_phonetic, matched_text
            )
        )
        print(f"lab_name:         {lab_name}")
        print(f"match_text:       {highlighted_text}")
        print(f"asr_result:       {highlighted_asr}")
        print(f"match_phonetic:   {highlighted_phonetic}")
        print(f"diff count:       {operation_count}")
        print("-" * 80)

    @staticmethod
    def _display_no_match(lab_name: str, asr_phonetic: List[str], reason: str = "") -> None:
        asr_str = " ".join(asr_phonetic)
        print(f"lab_name:         {lab_name}  -> 未能匹配到任何歌词片段")
        if reason:
            print(f"失败原因:         {reason}")
        print(f"asr_result (全部多余): {asr_str}")
        print("-" * 80)

    def execute(self) -> None:
        os.makedirs(self.json_folder, exist_ok=True)
        lyric_dict = self.load_all_lyrics()
        lab_pattern = f'{self.lab_folder}/*{self.LAB_EXTENSION}'
        asr_lab_files = glob.glob(lab_pattern)
        self.total_files = len(asr_lab_files)

        for lab_path in asr_lab_files:
            result = self.process_single_file(lab_path, lyric_dict)
            if result:
                self.compare_and_save_result(result)

        self.print_summary()
