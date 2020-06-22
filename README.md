# agwe-recipe

[TensorFlow Projector Demo](https://projector.tensorflow.org/?config=https://gist.githubusercontent.com/RuolinZheng08/ce93900f4876b63f598becfdc696f190/raw/073a0d87d441c7a58d34d7f21c8fc3c36b4e360e/projector-config.json)

Choose `Color by: suffix` to see words ending in "ing" colored in blue, words ending in "ly" colored in pink, and words ending in "tion" colored in red in 3D.

![Color by Suffix](https://github.com/RuolinZheng08/phonetic-acoustic-word-embeddings/blob/master/misc/suffix.gif)

![Demo](https://github.com/RuolinZheng08/phonetic-acoustic-word-embeddings/blob/master/misc/demo.gif)

This recipe trains acoustic word embeddings (AWEs) and acoustically grounded word embeddings (AGWEs) on paired data
consisting of word labels (given by their character sequences) and spoken word segments.

The training objective is based on the multiview triplet loss functions
of [Wanjia et al., 2016](https://arxiv.org/pdf/1611.04496.pdf).
Hard negative sampling was added in [Settle et al., 2019](https://arxiv.org/pdf/1903.12306.pdf) to improve
training speed (similar to `src/multiview_triplet_loss_old.py`). The current version (see `src/multiview_triplet_loss.py`) uses semi-hard negative sampling [Schroff et al.](https://arxiv.org/pdf/1503.03832.pdf) (instead of hard negative sampling) and includes `obj1` from Wanjia et al. in the loss.

### Dependencies
python 3.5+ (format string `f`), pytorch 1.4, h5py, numpy, scipy, [editdistance](https://github.com/roy-ht/editdistance/tree/master/editdistance)

### Dataset (for the purpose of TTIC 31110)
Use [this link](https://forms.gle/EGuaYYW72bzs4KbK8) to download the dataset.

### Training

Edit `train_config.json` and run `train.sh`
```
./train.sh
```
```json
"loss_objective": "obj0", or, "obj0+2"
"loss_edit_distance": null, or, "edit_distance", or, "weighted_edit_distance",
"loss_max_margin": 0.5,
"loss_max_threshold": 9,
"vocab_data_subwords": "phones", or, "words"
```

### Evaluate
Edit `eval_config.json` and run `eval.sh`
```
./eval.sh
```

### Results
With the default train_config.json you should obtain the following results: (`obj0+1+2`, `phones`, fixed margin)

acoustic_ap= 0.79

crossview_ap= 0.75

### Acknowledgement

This repo is forked from [Shane Settle's agwe-recipe repo](https://github.com/shane-settle/agwe-recipe).

