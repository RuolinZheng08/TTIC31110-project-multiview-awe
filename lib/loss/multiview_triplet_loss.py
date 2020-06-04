import logging as log

import torch
import torch.nn.functional as F


class MultiViewTripletLoss:

  def __init__(self, margin, k, min_k=None, extra=0, average=True):

    log.info(f" >> margin= {margin}")
    log.info(f" >> k= {k}")
    log.info(f" >> min_k= {min_k}")
    log.info(f" >> extra= {extra}")
    log.info(f" >> average= {average}")

    self.margin = margin
    self.k = k
    self.min_k = min_k or k
    self.extra = extra
    self.average = average

  def get_sims(self, x, y, inv, y_extra=None): # get_similarity

    n, d = x.shape
    m = y.shape[0]
    print('n, d, m', n, d, m)

    same = F.cosine_similarity(x, y[inv]).unsqueeze(-1)

    # shift y labels by one so now all x gets wrong labels
    perms = torch.cat([(inv + i) % m for i in range(1, m)])
    print('y', y)
    print('y[inv]', y[inv])
    print('inv', inv)
    print('perms', perms)
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
    # returns (top k values, indices)
    return x.topk(min(k, x.shape[dim]), dim=dim)

  def get_edistdist_tensor(self, true_label_ind, false_label_ind):
    # true_label_ind (batch_size, 1), false_label_ind (batch_size, k)
    # returns a tensor the same dimension as false_label_ind, i.e. (batch_size, k)
    editdist_tens = torch.zeros(false_label_ind.shape)
    # TODO: optimize tensor look up
    for true in true_label_ind:
      for false in false_label_ind:
        editdist_tens[true, false] = self.editdist_matrix[true, false]
    return editdist_tens

  def __call__(self, x, y, inv, y_extra=None):

    n, d = x.shape
    m = y.shape[0]

    if self.k > self.min_k:
      self.k -= 1

    k = min(self.k, m - 1)

    same, diff, perms = self.get_sims(x, y, inv, y_extra=y_extra) # same has dim (batch_size, 1)

    word_diff = self.get_word_sims(y, y_extra=y_extra)

    # NOTE: dis(a,b) = 1-cosine_similarity (see torch doc at https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.cosine_similarity)
    # Therefore, to add same (the dis(a,b) that's same across all objs),
    # it's equivalent to subtract that cosine_sim term.
    # The 1 in front cancels out with the diff dis term.

    # Most offending words per utt
    diff_k, diff_k_ind = self.get_topk(diff, k=k, dim=1) # diff_k has dim (batch_size, k)
    # TODO:
    if self.editdist_matrix is None:
      margin = self.margin # fixed margin
    else:
      editdist_tensor = self.get_editdist_tensor(inv, diff_k_ind)
      margin = self.margin_max * torch.min(self.threshold_max, editdist_tensor) / self.threshold_max
    obj0 = F.relu(margin + diff_k - same)

    # Most offending words per word
    word_diff_k, _ = self.get_topk(word_diff, k=k, dim=1)
    obj1 = F.relu(self.margin + word_diff_k[inv] - same)

    # Most offending utts per word
    utt_diff_k = torch.zeros(m, k, device=diff.device)
    for i in range(m):
      utt_diff_k[i], _ = self.get_topk(diff.view(-1)[perms == i], k=k)
    obj2 = F.relu(self.margin + utt_diff_k[inv] - same)

    # NOTE: this is modeled after obj1 but with x instead of y, so it might not be correct
    audio_diff = self.get_word_sims(x, y_extra=y_extra)
    audio_diff_k = self.get_topk(audio_diff, k=k, dim=1)
    obj3 = F.relu(self.margin + audio_diff_k[inv] - same)

    # TODO (cost-sensitive margins): obj0, obj1, obj2, obj3, obj0+2, obj1+3, obj0+1+2+3
    losses = {
      'obj0+2': (obj0 + obj2).mean(1), 
      'obj1+3': (obj1 + obj3).mean(1), # unsure
      'obj_all': (obj0 + obj1 + obj2 + obj3).mean(1) # unsure
    }

    # loss = (obj0 + obj1 + obj2).mean(1) # 1 is just the dim -- this is default with margin=0.5
    loss = losses['obj0+2']

    return loss.mean() if self.average else loss.sum()
