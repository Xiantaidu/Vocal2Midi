# LyricFA

## Intro

Using ASR to obtain syllables, matching text from lyrics, and generating JSON for Minlabel preloading.

[cpp version](https://github.com/openvpi/dataset-tools/releases/)

## How to use

1. Install

    ```bash
    pip install -r requirements.txt
    ```

2. Collect lyric
    1. Collect the original lyrics text and place it in the Lyric folder. The content is pure lyrics, and the file name
       is consistent with the audio before the AudioSlicer slicing (i.e. the part before the file name '_' after
       slicing)
       ```
       lyric
       ├── chuanqi.txt
       ├── caocao.txt
       └── ...
        ```

    2. Place the cut file fragments in the wav folder. Unify the file name with the previous lyrics: [lyricName]_
       xxx.wav.

       If there are multiple '_' in the file name, Take the far right as the dividing line. The file name in the left
       half must be the same as the lyrics file name in the previous step.
       ```
       wav
       ├── caocao_001.wav
       ├── caocao_002.wav
       └── ...
        ```

    3. Run fun_asr.py obtains the lab results of asr.
        ```
       python fun_asr.py --language zh/en --wav_folder wav_folder --lab_folder lab_folder
       
       Option:
           --language       str  zh/en
           --wav_folder     str  Sliced wav file folder (*.wav).
           --lab_folder     str  Folder for outputting lab files.       
       ```

    4. Run match_lyric.py obtains JSON and put it in the annotation folder of Minlabel.
       ```
       python match_lyric.py --lyric_folder lyric --lab_folder lab_folder --json_folder json_folder --language zh/en
       
       Option:
           --lyric_folder      str  The file name corresponds to the lab prefix (before \'_\'), only pure lyrics are allowed (*.txt).
           --lab_folder        str  Chinese characters or pinyin separated by spaces obtained from ASR (*.lab).
           --json_folder       str  Folder for outputting JSON files.
           --diff_threshold    int  Only display different results with n words or more.
           --language          str  zh/en
       ```

## Open-source softwares used

+ [zh_CN](https://github.com/ZiQiangWang/zh_CN)
  The core algorithm source has been further tailored to the dictionary in this project.

+ [RapidASR](https://github.com/RapidAI/RapidASR)
  The test data source.

+ [cc-edict](https://cc-cedict.org/wiki/)
  The dictionary source.

+ [mecab-python3](https://github.com/SamuraiT/mecab-python3)

+ [unidic-lite](https://github.com/polm/unidic-lite)

