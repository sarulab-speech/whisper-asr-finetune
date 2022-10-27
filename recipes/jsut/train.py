import yaml
from pathlib import Path 
import whisper
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import sys
sys.path.append(str(Path(__file__).resolve().absolute().parents[2]))
from whisper_finetune.dataset import load_data_list
from whisper_finetune.model import WhisperModelModule

# load config 
config_path = Path("config.yaml")
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

# dirs and paths
in_data_dir = Path(config["path"]["preprocessed"])
out_log_dir = Path(config["path"]["log"])
checkpoint_dir = Path(config["path"]["checkpoint"])
with_timestamps = bool(config["data"]["timestamps"])
train_config = config["train"]
device = "gpu" if torch.cuda.is_available() else "cpu"

out_log_dir.mkdir(parents=True, exist_ok=True)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

# tools
whisper_options = whisper.DecodingOptions(
    language=config["data"]["lang"], without_timestamps=not with_timestamps
)
whisper_tokenizer = whisper.tokenizer.get_tokenizer(
    True, language=config["data"]["lang"], task=whisper_options.task
)

# list
train_list = load_data_list(in_data_dir / "train.txt")
val_list = load_data_list(in_data_dir / "val.txt")

# logger
tflogger = TensorBoardLogger(
    save_dir=out_log_dir,
    name=config["train_name"],
    version=config["train_id"]
)

# callback
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir / "checkpoint",
    filename="checkpoint-{epoch:04d}",
    save_top_k=-1 # all model save
)
callback_list = [
    checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
model = WhisperModelModule(
    config["train"], config["model_name"], config["data"]["lang"], 
    train_list, val_list
)

trainer = Trainer(
    precision=16,
    accelerator=device,
    max_epochs=config["train"]["num_train_epochs"],
    accumulate_grad_batches=config["train"]["gradient_accumulation_steps"],
    logger=tflogger,
    callbacks=callback_list
)

trainer.fit(model)

