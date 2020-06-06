import logging as log
import json
import numpy as np
import torch

from data.datasets import Dataset


class MultiViewVocab(Dataset):

  def __init__(self, lexicon, subwords, counts=None, min_count=0):
    super(MultiViewVocab, self).__init__()

    with open(lexicon, "r") as f: # by default subwords = 'phones' from train/eval_config.json
      lexicon = json.load(f)
      w2i = lexicon["words_to_ids"]
      # NOTE: add 1 since padding_idx=0
      s2i = {s: i + 1 for s, i in lexicon[f"{subwords}_to_ids"].items()}
      # NOTE: use first pronunciation
      w2s = {w: s[0] for w, s in lexicon[f"word_to_{subwords}"].items()}

    num_removed = 0
    if min_count > 0:
      with open(counts, "r") as f:
        counts = json.load(f)
      for w in list(w2i):
        if counts.get(w, 0) < min_count:
          num_removed += 1
          del w2i[w], w2s[w]

    log.info(f" >> min_count= {min_count}")
    log.info(f" >> # words= {len(w2i)} (# removed= {num_removed})")
    log.info(f" >> # subwords= {len(s2i)}")
    for s, i in s2i.items():
      log.info(f"    {s}= {i}")

    # word to index, index to word, seq to index, word to seq
    self.w2i = w2i
    self.s2i = s2i
    self.w2s = w2s
    self.i2w = {i: w for w, i in w2i.items()}
    self.examples = list(w2i)

  def __getitem__(self, word):

    seq = np.array([self.s2i[s] for s in self.w2s[word]], dtype=np.int32)
    id_ = self.w2i[word]

    return {"seq": seq, "id": id_}

  def collate_fn(self, batch):

    lens = torch.LongTensor([len(ex["seq"]) for ex in batch])
    seqs = torch.zeros(len(lens), max(lens), dtype=torch.long)
    for i, ex in enumerate(batch):
        seqs[i, :lens[i]] = torch.from_numpy(ex["seq"])

    ids = np.array([ex["id"] for ex in batch], dtype=np.int32)

    return {"view2": seqs, "view2_lens": lens, "ids": ids}
