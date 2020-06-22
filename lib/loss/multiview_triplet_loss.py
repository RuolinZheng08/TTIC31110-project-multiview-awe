import logging as log

import torch
import torch.nn.functional as F

import editdistance
from .weighted_edit_distance import weighted_edit_distance

class MultiViewTripletLoss:

  def __init__(self, margin, k, max_margin, max_threshold, min_k=None, extra=0, 
  average=True, objective="obj0", edit_distance=None):

    log.info(f" >> margin= {margin}")
    log.info(f" >> k= {k}")
    log.info(f" >> max_margin= {max_margin}")
    log.info(f" >> max_threshold= {max_threshold}")
    log.info(f" >> min_k= {min_k}")
    log.info(f" >> extra= {extra}")
    log.info(f" >> average= {average}")
    log.info(f" >> objective= {objective}")
    log.info(f" >> edit_distance= {edit_distance}")

    self.margin = margin
    self.k = k
    self.min_k = min_k or k
    self.extra = extra
    self.average = average
    self.objective = objective

    # cost-sensitive margin with edit distance
    self.edit_distance = edit_distance
    self.max_margin = max_margin
    self.max_threshold = max_threshold

    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  def get_sims(self, x, y, inv, y_extra=None): # get_similarity

    n, d = x.shape
    m = y.shape[0]

    same = F.cosine_similarity(x, y[inv]).unsqueeze(-1)

    perms = torch.cat([(inv + i) % m for i in range(1, m)])
    diff = F.cosine_similarity(
        x.view(n, 1, d), y[perms].view(n, m - 1, d), dim=2)

    if y_extra is not None:
      diff_extra = F.cosine_similarity(
          x.view(n, 1, d), y_extra.view(1, -1, d), dim=2)
      diff = torch.cat([diff, diff_extra], dim=1)

    diff[diff > same] = -2.

    return same, diff, perms

  def get_word_sims(self, y, y_extra=None):

    m, d = y.shape
    m_range = torch.arange(m)

    rolls = torch.cat([torch.roll(m_range, i) for i in range(1, m)])
    word_diff = F.cosine_similarity(
        y.view(m, 1, d), y[rolls].view(m, m - 1, d), dim=2)

    if y_extra is not None:
      word_diff_extra = F.cosine_similarity(
          y.view(m, 1, d), y_extra.view(1, -1, d), dim=2)
      word_diff = torch.cat([word_diff, word_diff_extra], dim=1)

    word_diff[word_diff == 1.] = -2.

    return word_diff

  def get_topk(self, x, k, dim=0):

    return x.topk(min(k, x.shape[dim]), dim=dim) # torch fn that gets k biggest elts or something like that

  def get_editdist_tensor(self, true_label_ind, false_label_ind, eval_edit_dist_fn=editdistance.eval):
    # true_label_ind (batch_size, 1), false_label_ind (batch_size, k)
    # returns a tensor the same dimension as false_label_ind, i.e. (batch_size, k)
    # TODO: optimize tensor look up and check if it's correct
      editdist_tens = torch.empty(false_label_ind.shape, device=self.device)
      for true, i_true in enumerate(true_label_ind):
        seq_true = self.w2s[self.i2w[i_true.item()]]
        for k_false in false_label_ind:
          for false, i_false in enumerate(k_false):
            seq_false = self.w2s[self.i2w[i_false.item()]]
            dist = eval_edit_dist_fn(seq_true, seq_false)
            editdist_tens[true, false] = dist
      return editdist_tens

  def __call__(self, x, y, inv, y_extra=None):

    n, d = x.shape
    m = y.shape[0]

    if self.k > self.min_k:
      self.k -= 1

    k = min(self.k, m - 1)

    same, diff, perms = self.get_sims(x, y, inv, y_extra=y_extra)

    word_diff = self.get_word_sims(y, y_extra=y_extra)

    # NOTE: dis(a,b) = 1-cosine_similarity (see torch doc at https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.cosine_similarity)
    # Therefore, to add same (the dis(a,b) that's same across all objs),
    # it's equivalent to subtract that cosine_sim term.
    # The 1 in front cancels out with the diff dis term.

    # Most offending words per utt
    diff_k, diff_k_ind = self.get_topk(diff, k=k, dim=1) # diff_k has dim (batch_size, k)
    if self.edit_distance is None:
      obj0 = F.relu(self.margin + diff_k - same)
    elif self.edit_distance == "edit_distance":
      editdist_tensor = self.get_editdist_tensor(inv, diff_k_ind)
      margin = self.max_margin * torch.clamp(editdist_tensor, min=self.max_threshold) / self.max_threshold
      obj0 = F.relu(margin + diff_k - same)
    elif self.edit_distance == "weighted_edit_distance":
      editdist_tensor = self.get_editdist_tensor(inv, diff_k_ind, eval_edit_dist_fn=weighted_edit_distance)
      margin = self.max_margin * torch.clamp(editdist_tensor, min=self.max_threshold) / self.max_threshold
      obj0 = F.relu(margin + diff_k - same)

    # # Most offending words per word; used in 2019 paper but not in our proj
    # word_diff_k = self.get_topk(word_diff, k=k, dim=1)
    # obj1 = F.relu(self.margin + word_diff_k[inv] - same)

    # Most offending utts per word
    utt_diff_k = torch.zeros(m, k, device=diff.device)
    for i in range(m):
      utt_diff_k[i] = self.get_topk(diff.view(-1)[perms == i], k=k)
    obj2 = F.relu(self.margin + utt_diff_k[inv] - same)

    loss = obj0.mean(1)
    if self.objective == "obj0":
      loss = obj0.mean(1)
    elif self.objective == "obj0+2":
      loss = (obj0 + obj2).mean(1)

    return loss.mean() if self.average else loss.sum()
