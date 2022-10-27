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

