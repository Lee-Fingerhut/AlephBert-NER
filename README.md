# Named Entity Recognition for Hebrew using BERT

Contact: 
[Lee Fingerhut](mailto:leefingerhut@gmail.com)
[Peleg Zborovsky](mailto:peleg122@gmail.com)
[Tal Ben Gozi](mailto:talbg9@gmail.com)

## Installations Guide
1. Install an environment manager. Recommeneded: [Miniconda3](https://docs.conda.io/en/latest/miniconda.html).
   Here is a [Getting Started](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html#starting-conda) guide.
2. Clone the repo:
   ```sh
   git clone https://github.com/LeeFB/AlephBert-NER.git
   cd AlephBert-NER
   ```
4. Create a new environment from environment.yml (you can change the environment name in the file)
   ```sh
   conda env update -f environment.yml
   conda activate ner
   ```

## Training

```buildoutcfg
usage: ner_training.py [-h] [--seed SEED] [--name NAME] --train-file TRAIN_FILE [--max-seq-len MAX_SEQ_LEN] [--finetune]
                       [--num-epochs NUM_EPOCHS] [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE] [--optimizer-eps OPTIMIZER_EPS]
                       [--weight-decay-rate WEIGHT_DECAY_RATE] [--max-grad-norm MAX_GRAD_NORM] [--num-warmup-steps NUM_WARMUP_STEPS]

optional arguments:
  -h, --help            show this help message and exit

general:
  --seed SEED           seed for reproducibility
  --name NAME           name of directory for product

dataset:
  --train-file TRAIN_FILE
                        path to train file
  --max-seq-len MAX_SEQ_LEN
                        maximal sequence length

training:
  --num-epochs NUM_EPOCHS
                        number of epochs to train
  --batch-size BATCH_SIZE
                        batch size

optimizer:
  --learning-rate LEARNING_RATE
                        learning rate
  --optimizer-eps OPTIMIZER_EPS
                        optimizer tolerance
  --weight-decay-rate WEIGHT_DECAY_RATE
                        optimizer weight decay rate
  --max-grad-norm MAX_GRAD_NORM
                        maximal gradients norm

scheduler:
  --num-warmup-steps NUM_WARMUP_STEPS
                        scheduler warmup steps
```
BERT model is pretrained. \
You can enable all its parameters for training. \
Example:
```buildoutcfg
python ner_training --train-file dataset/cvs_data/spmrl/gold/morph_gold_train.csv --name sprml-train
```

## FineTuning 
BERT model is pretrained. \
you can freeze the encoder and finetune the classifier solely, by simply adding `--finetune` to training command. \
Example: 
```buildoutcfg
python ner_training --train-file dataset/cvs_data/spmrl/gold/morph_gold_train.csv --finetune --name sprml-finetune
```
