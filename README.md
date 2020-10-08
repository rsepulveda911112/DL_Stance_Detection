### Requirements

Install requirement file:
pip install -r requirement.txt

### Download the word embeddings

Download pretrained Wiki FastText Vectors

    wget https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
    mkdir -p data/fasttext
    unzip wiki.en.zip -d data/fasttext

Download pretrained GloVe Vectors

    wget http://nlp.stanford.edu/data/wordvecs/glove.6B.zip
    unzip glove.6B.zip -d data/glove
    gzip data/glove/*.txt  
    
### Download dataset
Use this git repository https://github.com/rsepulveda911112/FNC_Emergent_summary_dataset to download the datasets folder and copy in data folder

### Update configuration parameters
Update in src.utils.config.py the necessary parameters:

    
|Modes|Description|
|---|---|
| `training_set_file` | relative path of train dataset. |
| `test_set_file` | relative path of test dataset.|

   

### Run the end-to-end pipeline of the submitted models

    PYTHONPATH=src python src/stance_pipeline.py [--mode <mode>] [--dataset <dataset>]
All possible modes are as followings:

|Modes|Description|
|---|---|
| `train` | Train model. |
| `test` | Test model.|

All possible dataset are as followings:

|Dataset|Description|
|---|---|
| `fnc` | For use FNC dataset. |
| `emergent` | For use Emergent dataset.|
    
### Contacts:
If you have any questions please contact the authors. 
  * Robiert Sepúlveda Torres rsepulveda911112@gmail.com  
 
### License:
  * Apache License Version 2.0 