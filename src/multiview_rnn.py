import logging as log
import torch
import torch.nn as nn

from layer.rnn.rnn import RNN_default
from layer.linear import Linear
from saver.saver import Saver


class MultiViewRNN(nn.Module, Saver):

  def __init__(self,
               view1_cell,
               view1_input_size,
               view1_hidden_size,
               view1_num_layers,
               view1_bidirectional,
               view1_dropout,
               view1_proj,
               view2_cell,
               view2_num_embeddings,
               view2_input_size,
               view2_hidden_size,
               view2_num_layers,
               view2_bidirectional,
               view2_dropout,
               view2_proj,
               proj=None,
               loss_fn=None):
    nn.Module.__init__(self)
    Saver.__init__(self)

    self.net = nn.ModuleDict()

    log.info(f"view1:")
    self.net["view1"] = RNN_default(cell=view1_cell,
                                    input_size=view1_input_size,
                                    hidden_size=view1_hidden_size,
                                    num_layers=view1_num_layers,
                                    bidirectional=view1_bidirectional,
                                    dropout=view1_dropout,
                                    proj=view1_proj)

    log.info(f"view2:")
    self.net["view2"] = RNN_default(cell=view2_cell,
                                    num_embeddings=view2_num_embeddings,
                                    input_size=view2_input_size,
                                    hidden_size=view2_hidden_size,
                                    num_layers=view2_num_layers,
                                    bidirectional=view2_bidirectional,
                                    dropout=view2_dropout,
                                    proj=view2_proj)

    if proj is not None:
      log.info(f"proj:")
      self.net["proj"] = Linear(self.net["view1"].output_size, proj)

    if loss_fn is not None:
      self.loss_fn = loss_fn

    # TODO: pass editdist_matrix into loss_fn
    # self.loss_fn.editdist_matrix = self.editdist_matrix

  @property
  def output_size(self):
    if "proj" in self.net:
      return self.net["proj"].output_size
    else:
      return self.net["view1"].output_size

  def forward_view(self, batch, view):

    if view == "view1":
      if "starts" not in batch:
        _, _, emb = self.net["view1"](batch["view1"], batch["view1_lens"])
      else:
        emb = []
        out, lens, _ = self.net["view1"](batch["view1"], batch["view1_lens"])
        for i in range(len(out)):
          for j in range(len(batch["starts"][i])):
            start = batch["starts"][i][j]
            end = batch["ends"][i][j]
            emb.append(out[i, start:end + 1].mean(0))
        emb = torch.stack(emb)
    if view == "view2":
      _, _, emb = self.net["view2"](batch["view2"], batch["view2_lens"])

    if "proj" in self.net:
      emb = self.net["proj"](emb)

    return emb

  def forward(self, batch, view=None):
    h1 = self.forward_view(batch, "view1")
    h2 = self.forward_view(batch, "view2")
    return h1, h2

  def backward(self, loss, max_grad_norm=10.0):
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(
        self.parameters(), max_grad_norm)
    return loss.data.item(), grad_norm

  def train_step(self, batch, batch_iter):

    inv = batch.pop("inv")
    ids = batch.pop("ids")
    print('inv', inv, inv.shape)
    print('ids', ids, ids.shape)
    view1, view2 = self.forward(batch)

    loss = self.loss_fn(view1, view2, inv)
    loss_val, grad_norm = self.backward(loss)
    if not self.loss_fn.average:
      loss_val /= float(len(inv))

    log.info(f"{batch_iter}) "
             f"loss={loss_val:.3f} "
             f"grad_norm={grad_norm:.2f} "
             f"num_segments={len(inv)} "
             f"num_words={len(ids)}")
