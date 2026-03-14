import glob
import os
import time

import click
import librosa
from funasr import AutoModel


@click.command(help='ASR outputs lab annotations for multiple languages.')
@click.option('--language', type=click.Choice(['zh', 'en']), required=True,
              help='Language code: zh=Chinese, en=English')
@click.option('--wav_folder', required=True, metavar='Sliced wav file folder(*.wav).')
@click.option('--lab_folder', required=True, metavar='Folder for outputting lab files.')
def rapid_asr_multilingual(
        language: str = None,
        wav_folder: str = None,
        lab_folder: str = None
):
    assert wav_folder is not None and lab_folder is not None, 'wav input folder or lab output folder not entered.'
    os.makedirs(lab_folder, exist_ok=True)

    # Model mapping based on language
    model_mapping = {
        'zh': 'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
        'en': 'iic/speech_paraformer-large-vad-punc_asr_nat-en-16k-common-vocab10020'
    }

    # Load FunASR model
    model = AutoModel(
        model=model_mapping[language],
        model_revision="v2.0.4"
    )

    print(f"Started! Language: {language}")
    print("---------------")
    start_time = time.time()

    wav_list = glob.glob(os.path.join(wav_folder, '*.wav'))

    time_count = 0
    for wav_path in wav_list:
        time_count += librosa.get_duration(filename=wav_path)
        wav_name = os.path.splitext(os.path.basename(wav_path))[0]
        out_lab_path = os.path.join(lab_folder, f'{wav_name}.lab')

        if not os.path.exists(out_lab_path):
            try:
                # Load audio
                y, sr = librosa.load(wav_path, sr=16000, mono=True)

                # Run inference
                result = model.generate(
                    input=[y],  # Wrap in list for batch processing
                    cache={},
                    is_final=True
                )

                if result and len(result) > 0:
                    # Extract text from result
                    text = result[0].get('text', '') if isinstance(result[0], dict) else str(result[0])

                    print(f"{wav_path}\n{text}\n")

                    # Save to lab file
                    with open(out_lab_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                else:
                    print(f"{wav_path}: No result\n")

            except Exception as e:
                print(f"{wav_path}: Error - {str(e)}\n")
        else:
            print(f"{out_lab_path} exists, skip\n")

    end_time = time.time()
    elapsed_seconds = end_time - start_time
    print("---------------")
    print("Done!")
    if time_count != 0:
        print(f"RTF: {elapsed_seconds / time_count:.3f}x")
    print(f"Wav time: {time_count:.3f}s")
    print(f"Processing time: {elapsed_seconds:.3f}s")


if __name__ == '__main__':
    rapid_asr_multilingual()
