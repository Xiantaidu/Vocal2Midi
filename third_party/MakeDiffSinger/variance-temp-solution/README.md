# Making variance datasets (temporary solution)

This pipeline will guide you to migrate your old DiffSinger datasets to the new and complete format for both acoustic and variance model training.

## 1. Clone repo and install dependencies

```bash
git clone https://github.com/openvpi/MakeDiffSinger.git
cd MakeDiffSinger/variance-temp-solution
pip install -r requirements.txt  # or you can reuse a pre-existing DiffSinger environment
```

## 2. Convert transcriptions

Assume you have a DiffSinger dataset which contains a transcriptions.txt file.

Run:

```bash
python convert_txt.py path/to/your/transcriptions.txt
```

This will generate transcriptions.csv in the same folder as transcriptions.txt, which has three attributes: `name`, `ph_seq` and `ph_dur`.

## 3. Add `ph_num` attribute

The attribute `ph_num` is needed for training the variance models especially if you need to train the phoneme duration predictor. This attribute represents the number of phones that each word contains.

In singing, vowels, instead of consonants, are used to align with the beginnings of notes. For this reason, each word should start with a vowel/AP/SP, and end with leading consonant(s) of the next word (if there are any). See the example below:

```text
text      |   AP   |     shi     |        zhe       |  => word transcriptions (pinyin, romaji, etc.)
ph_seq    |   AP   |  sh  |  ir  | zh |      e      |  => phoneme sequence
ph_num    |       2       |     2     |      1      |  => word-level phoneme division
```

where `sh` and `zh` are consonants, `AP`, `ir` and `e` can be regarded as vowels. There are one special case that a word can start with a consonants: isolated consonants. In this case, all phones in the word are consonants.

For all monosyllabic phoneme systems (at most one vowel in one word), this step can be performed automatically.

### 3.1 two-part dictionaries (Chinese, Japanese, etc.)

A two-part dictionary has "V" and "C-V" phoneme patterns.

Run:

```bash
python add_ph_num.py path/to/your/transcriptions.csv --dictionary path/to/your/dictionary.txt
```

### 3.2 monosyllabic phoneme systems (Cantonese, Korean, etc.)

A universal monosyllabic phoneme system has "C(m)-V-C(n)" (m,n >= 0) phoneme patterns.

1. Collect all vowels into vowels.txt, divided by spaces.

2. Collect all consonants into consonants.txt, divided by spaces.

3. Run:

   ```bash
   python add_ph_num.py path/to/your/transcriptions.csv --vowels vowels.txt --consonants consonants.txt
   ```

### 3.3 polysyllabic phoneme systems (English, Russian, etc.)

We recommand this step be manually performed because word divisions cannot be infered from phoneme sequences in these phoneme systems.

> After finishing this step, the transcriptions.csv file can be directly used to train the phoneme duration predictor. If you want to train a pitch predictor, you must finish the remaining steps as follows.
>

## 4. Estimate note values

The note tier is another division of words besides the phoneme tier. See the example below:

```text
ph_seq       |   AP   |  sh  |  ir  | zh |      e      |  => phoneme sequence
ph_num       |       2       |     2     |      1      |  => word-level phoneme division
note_seq     |     rest      |    D#3    | D#3 |  C4   |  => note sequence
note_slur    |       0       |     0     |  0  |   1   |  => slur flag (will not be stored)
```

Note sequences can be automatically estimated and manually refined in two ways.

### 4.1 Infer a rough pitch value for each word

The following program can infer a rough note value for each word. There are no slurs - slurs are hard to judge, and different people have different labeling styles.

Run:

```bash
python estimate_midi.py path/to/your/transcriptions.csv path/to/your/wavs
```

> **IMPORTANT**
> 
> This step only estimates the rough MIDI value for each word. You have to refine the MIDI sequences, otherwise the pitch predictor will not be accurate.

### 4.2 (New!) Use the AI-powered MIDI extractor - SOME

SOME (Singing-Oriented MIDI Extractor) is a NN-based MIDI extractor developed under the DiffSinger ecosystem. See guidance [here](https://github.com/openvpi/SOME#inference-via-pretrained-model-diffsinger-dataset) for using it on your DiffSinger dataset.

## 5. Refine MIDI sequences

### 5.1 take apart transcriptions.csv into DS files

Run:

```bash
python convert_ds.py csv2ds path/to/your/transcriptions.csv path/to/your/wavs
```

This will generate *.ds files matching your *.wav files in the same directory.

> **IMPORTANT**
> 
> In this step, we highly recommend using RMVPE, a more accurate NN-based pitch extraction algorithm, to get better pitch results. See guidance [here](#rmvpe-pitch-extraction-algorithm).
> 
> Also note that after you finish manual MIDI refinement, please use the **same algorithm** and **same model** in your DiffSinger configuration files for variance model training to get the best results.

### 5.2 manually edit MIDI sequences

Get the latest release of SlurCutter from [here](../README.md#essential-tools-to-process-and-label-your-datasets). This simple tool helps you adjust MIDI pitch in each DS file and cut notes into slurs if neccessary. Be sure to back up your DS files before you start, since this tool will automatically save and overwrite an edited DS file.

### 5.3 re-combine DS files into transcriptions.csv

Run:

```bash
python convert_ds.py ds2csv path/to/your/ds path/to/your/transcriptions.csv
```

This will generate a new transcriptions.csv from the DS files you just edited. Append `-f` if you are sure you want to overwrite the original transcription file (and the script complains about it).

Now the transcriptions.csv can be used for all functionalities of DiffSinger training.

`convert_ds.py ds2csv` supports DS files which have no corresponding WAV files. All sentences in these files will be assigned a virtual item name, and inserted into the transcriptions. This is a preparation to support using DS tuning projects to train a variance model. In addition, `curves.json` file is written to support `f0` sequence refinement.

## (Appendix) other useful tools

### RMVPE pitch extraction algorithm

convert_ds.py and estimate_midi.py supports the state-of-the-art RMVPE pitch extraction algorithm. To use it:

- Install PyTorch via [official guidance](https://pytorch.org/get-started/locally/).
- Get RMVPE pretrained model [here](https://github.com/yxlllc/RMVPE/releases).
- Put the RMVPE model.pt in `variance-temp-solution/assets/rmvpe/`.
- Use `--pe rmvpe` when running `python convert_ds.py csv2ds` or `python estimate_midi.py`.

### correct_cents.py

Apply cents correction to note sequences in a transcriptions.csv to offset the out-of-tune errors. Need pitch extracted from waveforms for reference.

Usage:

```bash
python correct_cents.py csv path/to/your/transcriptions.csv path/to/your/wavs
```

or

```bash
python correct_cents.py ds path/to/your/ds/files
```

Note: this operation will overwrite your input file(s).

### eliminate_short.py

Eliminate short slur notes in DS files. Slurs that are shorter than a given threshold will be merged into its neighboring notes within the same word.

Usage:

```bash
python eliminate_short.py path/to/your/ds/files
```

Note: this operation will overwrite your input DS files.
