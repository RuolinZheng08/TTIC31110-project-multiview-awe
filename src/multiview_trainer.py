import logging as log

from utils.loader import load
from saver.saver import save_many, save_config, savez


def train(config):

  vocab = load("vocab_data", config)
  vocab_sampler = load("vocab_sampler", config, examples=vocab.examples)
  vocab.init_data_loader(vocab_sampler)

  train_data = load("train_data", config, vocab=vocab) # class multiview_dataset.MultiViewDataset_IndividualWords
  train_sampler = load("train_sampler", config, examples=train_data.examples)
  train_data.init_data_loader(train_sampler)

  epoch_len = len(train_data)
  eval_intervals = range(0, epoch_len, config.interval)

  dev_data = load("dev_data", config, vocab=vocab)
  dev_sampler = load("dev_sampler", config, examples=dev_data.examples)
  dev_data.init_data_loader(dev_sampler)

  test_data = load("test_data", config, vocab=vocab)
  test_sampler = load("test_sampler", config, examples=test_data.examples)
  test_data.init_data_loader(test_sampler)

  loss_fn = load("loss", config)
  eval_fn = load("eval", config)

  net = load("net", config, loss_fn=loss_fn,
             view1_input_size=train_data.input_feat_dim,
             view2_num_embeddings=train_data.input_num_subwords)
#   MultiViewRNN(
#   (net): ModuleDict(
#     (view1): RNN_default(
#       (rnn): GRU(108, 512, num_layers=6, batch_first=True, dropout=0.4, bidirectional=True)
#     )
#     (view2): RNN_default(
#       (emb): Embedding(46, 64, padding_idx=0)
#       (rnn): GRU(64, 512, num_layers=2, batch_first=True, dropout=0.4, bidirectional=True)
#     )
#   )
#   )
  net.cuda()

  optim = load("optim", config, params=net.parameters())
  sched = load("sched", config, optim=optim, net=net, eval_fn=eval_fn)

  if config.global_step > 0:
    train_data.load(tag=config.global_step)
    net.load(tag=config.global_step)
    optim.load(tag=config.global_step)
    sched.load(tag=config.global_step)

  while not optim.converged:

    log.info(f"epoch={config.global_step / epoch_len:.2f} "
             f"global_step={config.global_step} "
             f"lr={max(g['lr'] for g in optim.param_groups)} "
             f"start_iter={train_data.iter}")

    for i, batch in enumerate(train_data, train_data.iter):
      net.train()
      optim.zero_grad()
      net.train_step(batch, batch_iter=i)
      optim.step()

      config.global_step += 1

      if config.global_step % epoch_len in eval_intervals:
        log.info(f"epoch= {config.global_step / epoch_len:.2f}")
        log.info(f" >> time to evaluate and save:")
        save_many(train_data, net, optim, sched, tag=config.global_step)
        save_config(config)
        score, out = sched.eval_fn(net, dev_data)
        sched.step(config.global_step, score)

      if config.global_step == sched.best_global_step:
        save_many(train_data, net, optim, sched,
                  tag=config.global_step, best=True)

  log.info(f"Converged?= {optim.converged}")

  log.info("Evaluating best model on dev and test:")
  net.load(tag="best")

  log.info("dev score:")
  _, dev_out = eval_fn(net, dev_data)
  savez(f"save/dev-embs-{config.optim_lr}-{config.net_view1_dropout}-{config.loss_margin}", **dev_out)

  log.info(f"(This was for init_lr={config.optim_lr}, dropout={config.net_view1_dropout}, fixed_margin={config.loss_margin})")
  with open(f"dev-embs-{config.optim_lr}-{config.net_view1_dropout}-{config.loss_margin}.txt", 'a') as out:
    out.write(str(dev_out)+'\n')

  # commented out for now for tuning purposes!
  # log.info("test score:")
  # _, test_out = eval_fn(net, test_data)
  # savez("save/test-embs", **test_out)
