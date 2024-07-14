# Track1Foridash

## env build:

first create conda environment:
```
conda create -n DASH python=3.8.2
```
then use pip to build the environment
```
pip install -r DASHformer_environment.requirements
```

## evaluation
```
python DASHformer_Challenge.py -evaluate_DASHformer example_AA_sequences.list dashformer.keras dashformer_tokenizer.json 50 predictions.txt
```
get the result of all test data on the protein folder

## fine-tune

```
python DASHformer_Challenge.py -fine_tune example_AA_sequences.list <pretrain ckeckpoint> dashformer_tokenizer.json <max_seq_len> <epoch> <batchsize> <learning_rate>
```
finetuned weights can be found in the fine_tuned_model.keras in the CHALLENGE_DATA folder