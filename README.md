# DeepFam in Pytorch
PyTorch implementation of [DeepFam: deep learning based alignment-free method for protein family modeling and prediction](https://academic.oup.com/bioinformatics/article/34/13/i254/5045722), a deep learning based, alignment-free algorithm for protein family modeling. 

![Figure](https://github.com/bhi-kimlab/DeepFam/blob/master/docs/images/Figure1.png?raw=true)

## Prerequisites
- Python 3.6
- Pytorch 1.1.0

## Usage
Run the code with following commands.
```
python run.py --num_windows 256 256 256 256 256 256 256 256 \
--window_lengths 8 12 16 20 24 28 32 36 \
--train_file data/subfamily/train.txt \
--test_file data/subfamily/test.txt
```

To dedicate a target directory to write training logs, use `--log_dir` flag to designate the directory. 
