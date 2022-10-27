#from whisper_finetune.util import load_wav
#import tqdm
import yaml
from pathlib import Path 
import gdown
import shutil

# load config 
config_path = Path("config.yaml")
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

# download and unpack
corpus_url = "https://drive.google.com/uc?id=1f7bIQfwWdFOxeaYzs5Cw-HTcA8uwQ8qp"
out_zip_path = Path(config["path"]["download"]) / f"{config['corpus_name']}.zip"
out_zip_path.parent.mkdir(parents=True, exist_ok=True)

if not out_zip_path.exists():
    gdown.download(corpus_url, str(out_zip_path), quiet=False)
    shutil.unpack_archive(out_zip_path, out_zip_path.parent)
else:
    print("File exists; skip downloading corpus.")

# # copy
# transcript = {}
# data_dir = Path(config["path"]["data"])
# (data_dir / "wav").mkdir(parents=True, exist_ok=True)

# for transcript_path in (out_zip_path.parent / config['corpus_name']).glob("*/transcript_utf8.txt"):
#     for line in open(transcript_path, "r").readlines():
#         name, text = line.strip("\n").split(":")
#         in_wav_path = transcript_path.parent / "wav" / f"{name}.wav"

#         if not in_wav_path.exists():
#             continue
        
#         # copy wav
#         out_wav_path = data_dir / "wav" / in_wav_path.name
#         shutil.copy(in_wav_path, out_wav_path)

#         # store text
#         transcript[name] = text

# out_txt_path = data_dir / "transcript.txt"
# with open(out_txt_path, "w") as f:
#     f.writelines([f"{name}:{text}\n" for name, text in transcript.items()])




# def get_audio_file_list(
#     transcripts_path_list: list, text_max_length: int=120, audio_max_sample_length: int=480000, sample_rate: int=16000
#     ) -> list:
    
#     audio_transcript_pair_list = []
#     for transcripts_path in tqdm(transcripts_path_list):
#         # audioファイルのディレクトリ確認
#         audio_dir = transcripts_path.parent / "wav24kHz16bit"
#         if not audio_dir.exists():
#             print(f"{audio_dir}は存在しません。")
#             continue

#         # 翻訳テキストからAudioIdとテキストを取得
#         with open(transcripts_path, "r") as f:
#             text_list = f.readlines()
#         for text in text_list:
#             audio_id, text = text.replace("\n", "").split(":")
#             #print(audio_id, text)

#             audio_path = audio_dir / f"{audio_id}.wav"
#             if audio_path.exists():
#                 # データのチェック
#                 audio = load_wav(audio_path, sample_rate=sample_rate)[0]
#                 if len(text) > text_max_length or len(audio) > audio_max_sample_length:
#                     print(len(text), len(audio))
#                     continue
#                 audio_transcript_pair_list.append((audio_id, str(audio_path), text))
#     return audio_transcript_pair_list