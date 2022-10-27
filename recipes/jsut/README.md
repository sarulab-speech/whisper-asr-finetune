# Recipe for JSUT corpus
Finetuning the Whisper ASR model using the [JSUT corpus](https://sites.google.com/site/shinnosuketakamichi/publication/jsut). I recommend to use Anaconda or the other virtual enviroments.

## Step -1: Install
```
pip install -r requirement.txt
```

## Step 0: Configuration
See `config.yaml` to check the configurations. Switch data.frontend to `None` if you prefer to use raw text instead of Japanese Kana text.

## Step 1: Download
Download the corpus data and unzip.
```
python download.py
```

## Step 2: Preprocess
Extract features from audio and text.
```
python preprocessing.py
```

## Step 3: Training
Finetune the Whisper ASR model.
```
python train.py
```

## Step 4: Inference
Evaluate the finetuned model.
```
python inference.py
```
