# Hero name recognition

## Introduction
The project builds a siamese network to recognize the name of the hero image in Wild Rift.

## Training data
- Go to "https://leagueoflegends.fandom.com/wiki/League_of_Legends:_Wild_Rift" to download reference images for each champion.
- Search in Google "wild rift characters" to download more images.
- Create a folder for training data which we downloaded, it includes folders with folder names are champion names (see example in "collected_datatrain/data/").
- Structure of the datatest folder is similar with datatrain folder.

## Installation
- Local
```
conda create -n env_name python=3.8.18
conda activate env_name
pip install tqdm
pip install tensorflow==2.13.0
pip install opencv-python
pip install imutils
pip install tensorflow_addons
```
- Colab
```
pip install tensorflow_addons
```

## Prepare data
- Create training data: python create_datatrain.py "./collected_datatrain/data/" ./datatrain/imgs/ ./datatrain/ 50
    - "./collected_datatrain/data/": data which we collected for each champion
    - "./datatrain/imgs/": data which we augmented from collected data
    - "./datatrain/": to save pkl file for training data
    - 50: is the number of each champion wich we want to
- Create val data: python create_datatrain.py "./collected_datatrain/data/" ./dataval/imgs/ ./dataval/ 10
- Create testing data: python create_datatest.py "./datatest/ori/" ./datatest/
    - "./datatest/ori/": images of each champion for testing
    - "./datatest/": to save pkl file for testing data

## Training
- Adjust config at config.py such as datatrain path, batch size, num epoch
- python train.py

## Testing
- Adjust config at config.py such as ckpt path
- python test.py