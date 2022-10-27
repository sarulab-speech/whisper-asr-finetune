import torch
import torchaudio
import tqdm

def load_wav(wav_path, sample_rate:int=16000) -> torch.Tensor:
    """load audio file"""
    waveform, sr = torchaudio.load(wav_path, normalize=True)
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)
    return waveform


def load_txt(txt_path) -> str:
    txt = open(txt_path, "r").read()
    txt = txt.replace("\n", "")

    return txt


