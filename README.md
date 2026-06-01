# Vocal2Midi

Vocal2Midi 鏄竴涓鍒扮鐨勬瓕澹版帹鐞嗗伐鍏峰寘锛屽皢浜哄０闊抽杞崲涓哄甫鏈夋瓕璇嶅榻愮殑 MIDI 鏂囦欢銆? 
椤圭洰鍩轰簬 GAME 闊崇鎻愬彇妯″瀷鏋勫缓锛屽皢 ASR 璇煶璇嗗埆銆佸己鍒跺榻愬拰姝岃瘝鍖归厤闆嗘垚鍒颁竴涓畬鏁寸殑宸ヤ綔娴佷腑銆?

---

## 鐩綍

- [椤圭洰绠€浠媇(#椤圭洰绠€浠?
- [鏍稿績鐗规€(#鏍稿績鐗规€?
- [鏋舵瀯姒傝](#鏋舵瀯姒傝)
- [蹇€熷紑濮媇(#蹇€熷紑濮?
- [瀹夎鎸囧崡](#瀹夎鎸囧崡)
- [妯″瀷閰嶇疆](#妯″瀷閰嶇疆)
- [澶勭悊娴佺▼](#澶勭悊娴佺▼)
- [杈撳嚭鏍煎紡](#杈撳嚭鏍煎紡)
- [璇█鏀寔](#璇█鏀寔)
- [鏁呴殰鎺掗櫎](#鏁呴殰鎺掗櫎)
- [椤圭洰淇℃伅](#椤圭洰淇℃伅)

---

## 椤圭洰绠€浠?

Vocal2Midi 鎻愪緵浜嗕竴濂楀畬鏁寸殑姝屽０杞?MIDI 宸ュ叿閾撅細

| 缁勪欢 | 璇存槑 |
|------|------|
| **涓?GUI** | `app_fluent.py` 鈥?鍩轰簬 PyQt5 + qfluentwidgets 鐨勬闈㈢晫闈?|
| **涓荤绾?* | `inference/pipeline/auto_lyric_hybrid.py` 鈥?娣峰悎鎺ㄧ悊绠＄嚎 |
| **搴旂敤灞?* | `application/pipeline.py` 鈥?GUI 涓庢帹鐞嗘ā鍧椾箣闂寸殑缂栨帓杈圭晫 |

椤圭洰缁撴瀯閬靛惊鍒嗗眰鏋舵瀯锛歚gui 鈫?application 鈫?inference 鈫?modules/lib`

---

## 鏍稿績鐗规€?

- 姝屽０鍒?MIDI 鐨勫畬鏁存彁鍙栵紙闊崇鏃跺簭 + 闊抽珮锛?
- 鍩轰簬 Qwen3-ASR 鐨勮嚜鍔ㄦ瓕璇嶈浆褰曚笌瀵归綈
- 鍙€夌殑鍙傝€冩瓕璇嶄慨姝ｏ紙閫氳繃 LyricFA 鍖归厤锛?
- 涓枃锛坄zh`锛夊拰鏃ヨ锛坄ja`锛夋祦绋嬫敮鎸?
- 澶氱杈撳嚭鏍煎紡锛歚.mid`銆乣.ustx`銆乣.txt`銆乣.csv`銆乀extGrid
- RMVPE 闊抽珮鎻愬彇锛屾敮鎸佸甫闊抽珮鏇茬嚎锛坧itd锛夌殑 USTX 瀵煎嚭
- 鎵归噺澶勭悊
- 鍙腑鏂殑鍚庡彴浠诲姟锛屽疄鏃舵棩蹇楄緭鍑?

---

## 鏋舵瀯姒傝

```
gui/                鐣岄潰灞傦紙PyQt5 + qfluentwidgets锛?
application/        搴旂敤缂栨帓灞傦紙Use Case 鍏ュ彛锛?
inference/          鎺ㄧ悊涓庣畻娉曞疄鐜板眰
  鈹溾攢鈹€ pipeline/     娴佹按绾跨紪鎺?
  鈹溾攢鈹€ API/          鍚勬ā鍨?API锛圓SR / HFA / LFA / GAME / RMVPE / Slicer锛?
  鈹溾攢鈹€ io/           杈撳叆杈撳嚭涓庢牸寮忓鍑?
  鈹溾攢鈹€ slicer/       闊抽鍒囩墖
  鈹斺攢鈹€ quant/        闊崇閲忓寲
experiments/        妯″瀷瀛樻斁鐩綍
```

璇︾粏鏋舵瀯璇存槑瑙?[docs/architecture.md](docs/architecture.md)銆?

---

## 蹇€熷紑濮?

### Windows 鍚姩鍣紙鎺ㄨ崘锛?

鍙屽嚮杩愯锛?

```bat
start.bat
```

璇ヨ剼鏈細鎵ц `python app_fluent.py`锛岄粯璁や娇鐢ㄤ富 Fluent GUI 鍜屾贩鍚堢绾裤€?

### 鎵嬪姩鍚姩

涓?GUI锛?

```bash
python app_fluent.py
```

### 鍛戒护琛屾壒澶勭悊锛氬垏鐗?+ Qwen3-ASR / 鏁存 ASR

濡傛灉浣犲彧闇€瑕佲€滆緭鍏ヤ竴涓寘鍚?`.wav` 鐨勬枃浠跺す 鈫?鑷姩鍒囩墖 鈫?Qwen3-ASR 鈫?杈撳嚭鍒囩墖 wav 鍜?`.lab` 绾枃鏈€濓紝鍙互鐩存帴鐢細

```bash
python scripts/slice_asr_cli.py <input_dir> <output_dir> \
  --asr-model <Qwen3-ASR妯″瀷璺緞鎴朓D> \
  --device dml \
  --language zh \
  --slicing-method 榛樿鍒囩墖 \
  --asr-batch-size 4 \
  --file-batch-size 1
```

杈撳嚭缁撴瀯锛?

```text
output_dir/
  slices/<闊抽鍚?/...wav
  labs/<闊抽鍚?/...lab
```

璇存槑锛?

- `.lab` 鏂囦欢鍐呭鏄?ASR 璇嗗埆鍑虹殑绾枃鏈?- 鏀寔澶氭枃浠舵壒澶勭悊锛歚--file-batch-size`
- 鏀寔 ASR batch锛歚--asr-batch-size`
- 鍒囩墖绛栫暐娌跨敤椤圭洰鐜版湁鐨?`inference/API/slicer_api.py`

濡傛灉浣犺鏃ヨ鏁存璇嗗埆銆佷笉鍒囩墖锛岀洿鎺ュ姞 `--no-slice`锛?
```bash
python scripts/slice_asr_cli.py <input_dir> <output_dir> \
  --asr-model experiments/Qwen3-ASR-1.7B-dml \
  --device dml \
  --language ja \
  --no-slice \
  --asr-batch-size 4 \
  --file-batch-size 1
```

杩欎釜妯″紡浼氭妸姣忎釜鏂囦欢褰撴垚涓€涓畬鏁?chunk锛屼粛鐒惰緭鍑哄搴旂殑 `.wav` 鍜?`.lab`銆?
---

## 瀹夎鎸囧崡

### 鏂瑰紡涓€锛氳剼鏈畨瑁咃紙Windows锛?

```bat
瀹夎鐜.bat
```

### 鏂瑰紡浜岋細鎵嬪姩瀹夎

浣跨敤 Conda 鍒涘缓鐜锛?

```bash
conda env create -f environment.yml
conda activate vocal2midi_torch
```

鎴栦娇鐢?pip锛?

```bash
conda create -n vocal2midi python=3.12
conda activate vocal2midi
pip install -r requirements.txt
```

> Runtime note: `llama.cpp` stays on CPU; the ONNX backends default to DirectML and fall back to CPU when DML is unavailable.

---

## 妯″瀷閰嶇疆

杩愯鍓嶉渶瑕佸噯澶囦互涓嬫ā鍨嬭矾寰勶細

| 妯″瀷 | 璇存槑 |
|------|------|
| GAME 妯″瀷鐩綍 | 闊崇鎻愬彇鏍稿績妯″瀷 |
| HubertFA 妯″瀷鐩綍 | 闊崇礌寮哄埗瀵归綈 |
| ASR 妯″瀷璺緞 | Qwen3-ASR 鏈湴璺緞 |

寤鸿鐩綍缁撴瀯锛?

```text
experiments/
  GAME-1.0-medium/
  1218_hfa_model_new_dict/
  Qwen3-ASR-1.7B-dml/
  ...
```

鍦?GUI 鍏ㄥ眬璁剧疆涓厤缃ソ瀵瑰簲璺緞鍚庡啀杩愯闀挎椂闂翠换鍔°€?

---

## 澶勭悊娴佺▼

涓昏矾寰勶紙`app_fluent.py` 鈫?`auto_lyric_hybrid.py`锛夌殑澶勭悊娴佺▼濡備笅锛?

1. 鍔犺浇闊抽骞跺垏鍒嗕负鐗囨
2. 杩愯 ASR锛圦wen3-ASR锛夎繘琛岃闊宠瘑鍒?
3. 锛堝彲閫夛級灏?ASR 闊崇礌搴忓垪涓庡弬鑰冩瓕璇嶅尮閰嶏紙LyricFA锛?
4. 鐢熸垚 `.lab` 闊崇礌鏍囩鏂囦欢
5. 杩愯 HubertFA 寮哄埗瀵归綈
6. 杩愯 GAME 妯″瀷鎻愬彇闊崇涓庨煶楂?
7. 锛堝彲閫夛級杩愯 RMVPE 鎻愬彇甯х骇闊抽珮鏇茬嚎
8. 灏嗛煶绗︿笌姝岃瘝鍗曞厓瀵归綈
9. 瀵煎嚭鐩爣鏍煎紡锛坄.mid`銆乣.ustx` 绛夛級

---

## 杈撳嚭鏍煎紡

鏍规嵁閫夐」璁剧疆锛屽彲杈撳嚭浠ヤ笅鏍煎紡锛?

| 鏍煎紡 | 璇存槑 |
|------|------|
| `.mid` | 鏍囧噯 MIDI 鏂囦欢 |
| `.ustx` | OpenUtau 椤圭洰锛堝惈闊抽珮鏇茬嚎锛?|
| `.txt` | 闊崇/姝岃瘝鏂囨湰瀵煎嚭 |
| `.csv` | 琛ㄦ牸寮忛煶绗?姝岃瘝瀵煎嚭 |
| TextGrid | 璋冭瘯鎴栧垎鏋愮敤閫旂殑鏍囨敞鏂囦欢 |

ASR 涓庡尮閰嶈涓虹殑鏃ュ織涔熶細鍚屾鐢熸垚锛屼究浜庤拷韪洖閫€/鍛戒腑鐨勭墖娈点€?

---

## 璇█鏀寔

- 褰撳墠 GUI 鏀寔涓枃锛坄zh`锛夊拰鏃ヨ锛坄ja`锛夋ā寮?
- 鏃ヨ瀵归綈渚濊禆鏃ヨ G2P/鍒嗚瘝澶勭悊鍙?LyricFA 鍖归厤琛屼负
- 褰?ASR 缁撴灉瀛樺湪鏇挎崲鎴栨彃鍏ユ爣璁版椂锛屾彁渚涘弬鑰冩瓕璇嶅彲鏄捐憲鎻愬崌绋冲畾鎬?

---

## 鏁呴殰鎺掗櫎

| 闂 | 瑙ｅ喅鏂规 |
|------|----------|
| 鍚姩澶辫触 | 妫€鏌?Python 鐜鍜屼緷璧栨槸鍚﹀畨瑁呭畬鏁?|
| 妯″瀷鍔犺浇澶辫触 | 纭鎵€鏈夋ā鍨嬭矾寰勪负鏈夋晥鐩綍/鏂囦欢 |
| 鏃ヨ鍔熻兘寮傚父 | 瀹夎缂哄け鐨?G2P 渚濊禆鍖?|
| CUDA/鏄惧瓨涓嶈冻 | 鍦?GUI 璁剧疆涓噺灏?batch size |
| FFmpeg 缂哄け | 椤圭洰宸插唴缃?`_ffmpeg/bin/` 鐩綍 |

---

## 椤圭洰淇℃伅

- **璁稿彲璇?*: [MIT License](LICENSE)
- **鑷磋阿**: 瑙?[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md)
- **鏋舵瀯璇存槑**: 瑙?[docs/architecture.md](docs/architecture.md)




