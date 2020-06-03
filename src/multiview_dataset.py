import logging as log
import h5py
import numpy as np
import torch

from data.datasets import SpeechDataset


class MultiViewDataset(SpeechDataset):

  def __init__(self, feats, align, vocab, min_dur=2, max_dur=None,
               min_seg_dur=2, max_seg_dur=None, stack=False):
    super(MultiViewDataset, self).__init__()

    feats = h5py.File(feats, "r")
    align = h5py.File(align, "r")

    num_ignored = 0
    examples = {}
    for uid, g in align.items():
      ind = []
      frames = len(feats[uid][:])
      if frames < min_dur or (max_dur is not None and frames > max_dur):
        continue
      words = g["words"][:]
      durs = g["ends"][:] - g["starts"][:] + 1
      for i, (w, d) in enumerate(zip(words, durs)):
        if d < min_seg_dur or (max_seg_dur is not None and d > max_seg_dur):
          num_ignored += 1
          continue
        if w not in vocab.w2i:
          continue
        ind.append(i)
      if len(ind) == 0:
        num_ignored += 1
        continue
      examples[(uid, tuple(ind))] = {
        "frames": frames // 2 if stack else frames,
        "words": len(ind)
      }

    seg = self.add_deltas(feats[list(examples)[0][0]][:])
    if stack:
      seg = self.stack_input_frames(seg)

    _, input_feat_dim = seg.shape
    input_num_subwords = len(vocab.s2i)

    log.info(f" >> feats: loading {feats.filename}")
    log.info(f" >> align: loading {align.filename}")
    log.info(f" >> stack= {stack}")
    log.info(f" >> durs= ({min_dur}, {max_dur})")
    log.info(f" >> seg_durs= ({min_seg_dur}, {max_seg_dur})")
    log.info(f" >> # utts= {len(examples)} (# ignored= {num_ignored})")
    log.info(f" >> # words= {sum(len(k[1]) for k in examples)}")
    log.info(f" >> input_feat_dim= {input_feat_dim}")
    log.info(f" >> input_num_subwords= {input_num_subwords}")

    self.feats = feats
    self.stack = stack
    self.align = align
    self.vocab = vocab
    self.examples = examples
    self.input_feat_dim = input_feat_dim
    self.input_num_subwords = input_num_subwords

  def __getitem__(self, ex):

    uid, ind = ex

    seg = self.feats[uid][:]
    starts = [self.align[uid]["starts"][:][i] for i in ind]
    ends = [self.align[uid]["ends"][:][i] for i in ind]
    seg = self.add_deltas(seg)
    if self.stack:
      seg = self.stack_input_frames(seg)
      starts = [start // 2 for start in starts]
      ends = [end // 2 for end in ends]

    words = np.array([self.align[uid]["words"][:][i] for i in ind])

    return {
      "seg": seg, "starts": starts, "ends": ends,
      **self.vocab.collate_fn([self.vocab[word] for word in words])
    }


class MultiViewDataset_Contextual(MultiViewDataset):

  def collate_fn(self, batch):

    durs = torch.LongTensor([len(ex["seg"]) for ex in batch])
    segs = torch.zeros(len(durs), max(durs), self.input_feat_dim)
    starts, ends = [], []
    for i, ex in enumerate(batch):
      segs[i, :durs[i]] = torch.from_numpy(ex["seg"])
      starts.append(ex["starts"])
      ends.append(ex["ends"])

    view2_lens = torch.cat([ex["view2_lens"] for ex in batch])
    view2 = torch.zeros(len(view2_lens), max(view2_lens), dtype=torch.long)
    j = 0
    for i, ex in enumerate(batch):
      view2[j:j + len(ex["view2"]), :max(ex["view2_lens"])] = ex["view2"]
      j += len(ex["view2"])

    ids = np.hstack([ex["ids"] for ex in batch])
    uids, ind, inv_ind = np.unique(
        ids, return_index=True, return_inverse=True)

    return {
      "view1": segs, "view1_lens": durs, "starts": starts, "ends": ends,
      "view2": view2[ind], "view2_lens": view2_lens[ind],
      "ids": uids, "inv": torch.from_numpy(inv_ind)
    }


class MultiViewDataset_Isolated(MultiViewDataset):

  def collate_fn(self, batch):

    seg_list = []
    for ex in batch:
      for start, end in zip(ex["starts"], ex["ends"]):
        seg_list.append(ex["seg"][start:end + 1])

    durs = torch.LongTensor([len(seg) for seg in seg_list])
    segs = torch.zeros(len(durs), max(durs), self.input_feat_dim)
    for i, seg in enumerate(seg_list):
      segs[i, :durs[i]] = torch.from_numpy(seg)

    view2_lens = torch.cat([ex["view2_lens"] for ex in batch])
    view2 = torch.zeros(len(view2_lens), max(view2_lens), dtype=torch.long)
    j = 0
    for i, ex in enumerate(batch):
      view2[j:j + len(ex["view2"]), :max(ex["view2_lens"])] = ex["view2"]
      j += len(ex["view2"])

    ids = np.hstack([ex["ids"] for ex in batch])
    uids, ind, inv_ind = np.unique(
        ids, return_index=True, return_inverse=True)

    return {
      "view1": segs, "view1_lens": durs,
      "view2": view2[ind], "view2_lens": view2_lens[ind],
      "ids": uids, "inv": torch.from_numpy(inv_ind)
    }


class MultiViewDataset_IndividualWords(SpeechDataset):

  def __init__(self, feats, align, vocab,
               min_seg_dur=2, max_seg_dur=None, stack=False):
    super(MultiViewDataset_IndividualWords, self).__init__()

    feats = h5py.File(feats, "r")
    align = h5py.File(align, "r")

    num_ignored = 0
    examples = {}
    for uid, g in align.items():
      words = g["words"][:] # list of word(s)
      durs = g["ends"][:] - g["starts"][:] + 1
      for i, (w, d) in enumerate(zip(words, durs)):
        if d < min_seg_dur or (max_seg_dur is not None and d > max_seg_dur):
          num_ignored += 1
          continue
        if w not in vocab.w2i:
          continue
        examples[(uid, i)] = {"frames": d}

    seg = self.add_deltas(feats[list(examples)[0][0]][:])
    if stack:
      seg = self.stack_input_frames(seg)

    _, input_feat_dim = seg.shape
    input_num_subwords = len(vocab.s2i)

    log.info(f" >> feats: loading {feats.filename}")
    log.info(f" >> align: loading {align.filename}")
    log.info(f" >> stack= {stack}")
    log.info(f" >> seg_durs= ({min_seg_dur}, {max_seg_dur})")
    log.info(f" >> # words= {len(examples)} (# ignored= {num_ignored})")
    log.info(f" >> input_feat_dim= {input_feat_dim}")
    log.info(f" >> input_num_subwords= {input_num_subwords}")

    self.feats = feats
    self.stack = stack
    self.align = align
    self.vocab = vocab
    self.examples = examples
    self.input_feat_dim = input_feat_dim
    self.input_num_subwords = input_num_subwords

  def __getitem__(self, ex):

    uid, i = ex

    start = self.align[uid]["starts"][:][i]
    end = self.align[uid]["ends"][:][i]
    seg = self.feats[uid][:][start:end]
    seg = self.add_deltas(seg)
    if self.stack:
      seg = self.stack_input_frames(seg)

    word = self.align[uid]["words"][:][i]

    return {"seg": seg, **self.vocab[word]}

  def collate_fn(self, batch):

    durs = torch.LongTensor([len(ex["seg"]) for ex in batch])
    segs = torch.zeros(len(durs), max(durs), self.input_feat_dim)
    for i, ex in enumerate(batch):
      segs[i, :durs[i]] = torch.from_numpy(ex["seg"])

    ids = np.array([ex["id"] for ex in batch], dtype=np.int32)
    uids, ind, inv_ind = np.unique(
        ids, return_index=True, return_inverse=True)

    lens = torch.LongTensor([len(batch[j]["seq"]) for j in ind])
    seqs = torch.zeros(len(lens), max(lens), dtype=torch.long)
    for i, j in enumerate(ind):
      seqs[i, :lens[i]] = torch.from_numpy(batch[j]["seq"])

    return {
      "view1": segs, "view1_lens": durs,
      "view2": seqs, "view2_lens": lens,
      "ids": uids, "inv": torch.from_numpy(inv_ind)
    }
