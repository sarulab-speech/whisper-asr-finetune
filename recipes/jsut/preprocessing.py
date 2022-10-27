#from whisper_finetune.util import load_wav
#import tqdm
import yaml
from pathlib import Path 
import whisper
import numpy as np 
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(Path(__file__).resolve().absolute().parents[2]))
from whisper_finetune.dataset import valid_audio_text_safe, save_data_list
from whisper_finetune.util import load_wav
from whisper_finetune.frontend import TextFrontend

# load config 
config_path = Path("config.yaml")
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
out_zip_path = Path(config["path"]["download"]) / f"{config['corpus_name']}.zip"
text_max_length = config["data"]["text_max_length"]
audio_max_length = config["data"]["audio_max_length"]
audio_sampling_rate = config["data"]["audio_sampling_rate"]

train_ratio = config["data"]["train_ratio"]
val_ratio = config["data"]["val_ratio"]
test_ratio = 1 - train_ratio - val_ratio
assert test_ratio > 0

# tool
frontend = TextFrontend(config["data"]["frontend"])

# make data list
data_list = []
data_dir = Path(config["path"]["preprocessed"])
(data_dir / "mel").mkdir(parents=True, exist_ok=True)

for transcript_path in (out_zip_path.parent / config['corpus_name']).glob("*/transcript_utf8.txt"):
    for line in tqdm(open(transcript_path, "r").readlines(), desc=f"processing {transcript_path.parent}"):
        # text frontend
        name, text = line.strip("\n").split(":")
        text = frontend(text).replace("\t", "")

        # audio frontend
        in_wav_path = transcript_path.parent / "wav" / f"{name}.wav"
        wav = load_wav(in_wav_path)

        if not valid_audio_text_safe(text, wav, text_max_length, audio_max_length):
            continue

        # wav -> mel
        wav = whisper.pad_or_trim(wav.flatten())
        mel = whisper.log_mel_spectrogram(wav).numpy().astype(np.float32)
        out_mel_path = data_dir / "mel" / in_wav_path.with_suffix(".npy").name
        np.save(out_mel_path, mel)

        # add to list
        data_list.append([name, str(out_mel_path), text])
    break

# save list
train_list, val_test_list = train_test_split(data_list, train_size=train_ratio)
val_list, test_list = train_test_split(val_test_list, train_size=val_ratio/(val_ratio + test_ratio))

for name, lst in zip(["train", "val", "test"], [train_list, val_list, test_list]):
    save_data_list(lst, data_dir / f"{name}.txt")
