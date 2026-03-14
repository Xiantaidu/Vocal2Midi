# Making Datasets from Scratch (Forced Alignment)

This pipeline will guide you to build your dataset from raw recordings with MFA (Montreal Forced Aligner).

## 0. Requirements

This pipeline will require your dictionary having its corresponding MFA pretrained model. You can see currently supported dictionaries and download their MFA models in the table below:

|  dictionary name   |    dictionary file     |                                          MFA model                                           |
|:------------------:|:----------------------:|:--------------------------------------------------------------------------------------------:|
| Opencpop extension | opencpop-extension.txt | [link](https://huggingface.co/datasets/fox7005/tool/resolve/main/mfa-opencpop-extension.zip) |

Your recordings must meet the following conditions:

1. They must be in one single folder. Files in sub-folders will be ignored.
2. They must be in WAV format.
3. They must have a sampling rate higher than 32 kHz.
4. They should be clean, unaccompanied voices with no significant noise or reverb.
5. They should contain only voices from one single human.

<font color="red">**NOTICE:**</font> Before you train a model, you must obtain permission from the copyright holder of the dataset and make sure the provider is fully aware that you will train a model from their data, that you will or will not distribute the synthesized voices and model weights, and the potential risks of this kind of activity.

## 1. Clone repo and install dependencies

```bash
git clone https://github.com/openvpi/MakeDiffSinger.git
cd MakeDiffSinger/acoustic-forced-alignment
conda create -n mfa python=3.8 --yes  # you must use a Conda environment!
conda activate mfa
conda install -c conda-forge montreal-forced-aligner==2.0.6 --yes  # install MFA
pip install -r requirements.txt  # install other requirements
```

## 2. Prepare recordings and transcriptions

### 2.1 Audio slicing

The raw data must be sliced into segments of about 5-15 seconds. We recommend using [AudioSlicer](../README.md#essential-tools-to-process-and-label-your-datasets), a simple GUI application that can automatically slice audio files via silence detection.

Run the following command to validate your segment lengths and count the total length of your sliced segments:

```bash
python validate_lengths.py --dir path/to/your/segments/
```

### 2.2 Label the segments

All segments should have their transcriptions (or lyrics) annotated. See [assets/2001000001.wav](assets/2001000001.wav) and its corresponding label [assets/2001000001.lab](assets/2001000001.lab) as an example.

Each segment should have one annotation file with the same filename as it and `.lab` extension, and placed in the same directory. In the annotation file, you should write all syllables sung or spoken in this segment. Syllables should be split by space, and only syllables that appears in the dictionary are allowed. In addition, all phonemes in the dictionary should be covered in the annotations. Please note that `SP`, `AP` and `<PAD>` should not be included in the labels although they are in your final phoneme set.

We developed [MinLabel](../README.md#essential-tools-to-process-and-label-your-datasets), a simple yet efficient tool to help finishing this step.

Once you finish labeling, run the following command to validate your labels:

```bash
python validate_labels.py --dir path/to/your/segments/ --dictionary path/to/your/dictionary.txt
```

This will ensure:

- All recordings have their corresponding labels.
- There are no unrecognizable phonemes that does not appear in the dictionary.
- All phonemes in the dictionary are covered by the labels.

If there are failed checks, please fix them and run again.

A summary of your phoneme coverage will be generated. If there are some phonemes that have extremely few occurrences (for example, less than 20), it is highly recommended to add more recordings to cover these phonemes.

## 3. Forced Alignment

### 3.1 Reformat recordings

Given the transcriptions of each segment, we are able to align the phoneme sequence to its corresponding audio, thus obtaining position and duration information of each phoneme.

We use [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) to do forced phoneme alignment.

MFA fails on some platforms if the WAVs are not in 16kHz 16bit PCM format. The following command will reformat your recordings and copy the labels to another temporary directory. You may delete those temporary files afterwards.

```bash
python reformat_wavs.py --src path/to/your/segments/ --dst path/to/tmp/dir/
```

NOTE: `--normalize` can be added to normalize the audio files with respect to the peak value of the whole segments. This is especially helpful on aspiration detection during TextGrid enhancement if the original segments are too quite.

### 3.2 Run MFA on the corpus

MFA will align your labels to your recordings and save the results to TextGrid files.

Download the MFA model and run the following command:

```bash
mfa align path/to/your/segments/ path/to/your/dictionary.txt path/to/your/model.zip path/to/your/textgrids/ --beam 100 --clean --overwrite
```

Run the following command to check if all TextGrids are successfully generated:

```bash
python check_tg.py --wavs path/to/your/segments/ --tg path/to/your/textgrids/
```

If the checks above fails, or the results are not good, please try another `--beam` value and run the MFA again. TextGrids generated by MFA are still raw and need further processing, so please do not edit them at this time.

### 3.3 Enhance and finish the TextGrids

MFA results might not be good on some long utterances. In this section, we:

- Try to reduce errors for long utterances
- Detect `AP`s and add `SP`s which have not been labeled before.

Run:

```bash
python enhance_tg.py --wavs path/to/your/segments/ --dictionary path/to/your/dictionary.txt --src path/to/raw/textgrids/ --dst path/to/final/textgrids/
```

NOTE: There are other useful arguments of this script. If you understand them, you can try to get better results through adjusting those parameters.

The final TextGrids can be saved for future use.

If you are interested in the word-level pitch distribution of your dataset, run the following command:

```bash
python summary_pitch.py --wavs path/to/your/segments/ --tg path/to/final/textgrids/
```

### 3.4 (Optional) Manual TextGrids refinement

With steps above, the TextGrids we get contains 2 tiers: the words and the phones. Manual refinement to your TextGrids may take lots of effort but will boost the performance and stability of your model.

This section is a recommended (but not required) way to refine your TextGrids manually. Before you start, an additional dependency to achieve natural sorting needs to be installed:

```bash
pip install natsort
```

#### 3.4.1 Combine the recordings and TextGrids

A full dataset can contain hundreds or thousands of auto-sliced recording segments and their corresponding TextGrids. The following command will combine them into long ones:

```bash
python combine_tg.py --wavs path/to/your/segments/ --tg path/to/your/final/textgrids/ --out path/to/your/combined/textgrids/
```

This will combine all items with same name except their suffixes and add a `sentences` tier in the combined TextGrids. The new sentences tier controls how the long combined recordings are split into short sentences. If you have other suffix pattern (default: `"_\d+"`) or want to change the bit-depth (default: PCM_16) of the combined recordings, see `python combine_tg.py --help`.

#### 3.4.2 Manual editing

TextGrids can be viewed and edited with [Praat](https://github.com/praat/praat) or [vLabeler](https://github.com/sdercolin/vlabeler) (recommended).

The editing mainly involves the sentences tier and the phones tier. When editing, please ensure the sentences tier is aligned with the words and phones tier; but it is not required to align the words tier to the phones tier. If you want to remove a sentence or not to include one area in any sentences, just leave an empty mark on that area.

#### 3.4.3 Slice the recordings and TextGrids

After manual editing is finished, the words tier can be automatically re-aligned to the phones tier. Run:

```bash
python align_tg_words.py --tg path/to/your/combined/textgrids --dictionary path/to/your/dictionary.txt --overwrite
```

NOTE 1: This will overwrite your TextGrid files. You can back them up before running the command, or specify another output directory with `--out` option.

NOTE 2: This script is also compatible with segmented 2-tier TextGrids.

Then the TextGrids and recordings can be sliced according to the boundaries stored in the sentences tiers. Run:

```bash
python slice_tg.py --wavs path/to/your/combined/textgrids/ --out path/to/your/sliced/textgrids/refined/
```

By default, the output segments will be re-numbered like `item_000`, `item_001`, ..., `item_XXX`. If you want to use the marks stored in the sentences tier as the filenames, or want to change the bit-depth (default: PCM_16) of the sliced recordings, or control other behaviors, see `python slice_tg.py --help`.

Now you can use these manually refined and re-sliced TextGrids and recordings for further steps.

## 4. Build the final dataset

The TextGrids need to be collected into a transcriptions.csv file as the final transcriptions. The CSV file will include the following columns:

- name: the segment name
- ph_seq: the phoneme sequence
- ph_dur: the phoneme duration

The recordings will be arranged like [this](../README.md#diffsinger-dataset-structure).

Run:

```bash
python build_dataset.py --wavs path/to/your/segments/ --tg path/to/final/textgrids/ --dataset path/to/your/dataset/
```

NOTE 1: This will insert random silence parts around each segments by default for better `SP` stability. If you do not need these silence parts, for example, if your TextGrids have been manually refined, please use the `--skip_silence_insertion` option.

NOTE 2: `--wav_subtype` can be used to specify the bit-depth of the saved WAV files. Options are `PCM_16` (default), `PCM_24`, `PCM_32`, `FLOAT`, and `DOUBLE`.

After doing all things above, you should put it into data/ of the DiffSinger main repository. Now, your dataset can be used to train DiffSinger acoustic models. If you want to train DiffSinger variance models, please follow instructions [here](../variance-temp-solution/README.md).

## 5. Write configuration file

Copy the template configration file from `configs/templates` in the DiffSinger repository to your data folder, or a new folder if working with multi-speaker model. Specify required fields in the configurations, check `DiffSinger/docs/ConfigurationSchemas.md` for help on the meanings of those fields.

For automatic validation set selection, you can leave the following field as empty. If the field is not empty, the script will prompt a overwrite confirmation later.
```yaml
...
test_prefixes:
...
```

And run:
```bash
python select_test_set.py path/to/your/config.yaml [--rel_path <PATH>]
```

NOTE 1: `--rel_path` is probably necessary if there are relative paths in your config file. If only absolute paths exist in it, you can omit this argument.

NOTE 2: There are other useful arguments of this script. You can use them to change the total number of validation samples.
