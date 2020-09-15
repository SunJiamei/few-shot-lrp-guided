import numpy as np
import os
import random
import torch
from data.datamgr import SetDataManager, SimpleDataManager
from options import parse_args, get_resume_file, load_warmup_state
from methods.LFTNet import LFTNet
import utils
def cycle(iterable):
  while True:
    for x in iterable:
      yield x

# training iterations
def train(base_datamgr, base_set, aux_iter, val_loader, model, start_epoch, stop_epoch, params):

  # for validation
  max_acc = 0
  total_it = 0
  # training
  for epoch in range(start_epoch,stop_epoch):
    if params.adj_lr==True:
      learning_rate_adj = params.LUT_lr
      model_lr = utils.adjust_learning_rate(model.model_optim, epoch, learning_rate_adj)
      ft_lr = utils.adjust_learning_rate(model.ft_optim, epoch, learning_rate_adj)
    # randomly split seen domains to pseudo-seen and pseudo-unseen domains
    random_set = random.sample(base_set, k=2)
    ps_set = random_set[0]
    pu_set = random_set[1:]
    ps_loader = base_datamgr.get_data_loader(os.path.join(params.data_dir, ps_set, 'base.json'), aug=params.train_aug)
    pu_loader = base_datamgr.get_data_loader([os.path.join(params.data_dir, dataset, 'base.json') for dataset in pu_set], aug=params.train_aug)
    base_loader = base_datamgr.get_data_loader([os.path.join(params.data_dir, dataset, 'base.json') for dataset in base_set], aug=params.train_aug)
    # train loop
    model.train()
    if params.feature_wise_type == 'FT':
      total_it = model.train_loop(epoch, base_loader, total_it)
    else:
      total_it = model.trainall_loop(epoch, ps_loader, pu_loader, aux_iter, total_it)

    # validate
    model.eval()
    with torch.no_grad():
      acc = model.test_loop(val_loader)

    # save
    if acc > max_acc:
      print(f"best model! accuracy: {acc}, save...")
      max_acc = acc
      outfile = os.path.join(params.checkpoint_dir, 'best_model.tar')
      model.save(outfile, epoch)
    else:
      print('GG!! best accuracy {:f}'.format(max_acc))
    if ((epoch + 1) % params.save_freq==0) or (epoch == stop_epoch - 1):
      outfile = os.path.join(params.checkpoint_dir, '{:d}.tar'.format(epoch + 1))
      model.save(outfile, epoch)

  return


# --- main function ---
if __name__=='__main__':

  # set numpy random seed
  np.random.seed(1)

  # parse argument
  params = parse_args('train')


  print(params)

  # output and tensorboard dir
  params.tf_dir = '%s/log/%s'%(params.save_dir, params.name)
  params.checkpoint_dir = '%s/checkpoints/%s'%(params.save_dir, params.name)
  if not os.path.isdir(params.checkpoint_dir):
    os.makedirs(params.checkpoint_dir)

  # dataloader
  print('\n--- prepare dataloader ---')
  print('  train with multiple seen domains (unseen domain: {})'.format(params.testset))
  datasets = ['miniImagenet', 'cars', 'places', 'cub', 'plantae']
  datasets.remove(params.testset)
  val_file = os.path.join(params.data_dir, 'miniImagenet', 'val.json')

  # model
  print('\n--- build LFTNet model ---')
  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224

  n_query = max(1, int(16* params.test_n_way/params.train_n_way))
  train_few_shot_params   = dict(n_way = params.train_n_way, n_support = params.n_shot)
  base_datamgr            = SetDataManager(image_size, n_query = n_query,  **train_few_shot_params)
  aux_datamgr             = SimpleDataManager(image_size, batch_size=16)
  aux_iter              = iter(cycle(aux_datamgr.get_data_loader(os.path.join(params.data_dir, 'miniImagenet', 'base.json'), aug=params.train_aug)))
  test_few_shot_params    = dict(n_way = params.test_n_way, n_support = params.n_shot)
  val_datamgr             = SetDataManager(image_size, n_query = n_query, **test_few_shot_params)
  val_loader              = val_datamgr.get_data_loader( val_file, aug = False)

  if params.feature_wise_type  in ['LFT', 'FT']:
    print('--- LFTNet training: {} ---\n'.format(params.name))
    model = LFTNet(params, tf_path=params.tf_dir)
  elif params.feature_wise_type in ['LFTAdaIN']:
    print('--- LFTAdaINNet training: {} ---\n'.format(params.name))
    model = LFTAdaINNet(params, tf_path=params.tf_dir)
  else:
    raise NotImplementedError('please specify the feature_wise_type when training with LFT')
  model.cuda()

  # resume training
  start_epoch = params.start_epoch
  stop_epoch = params.stop_epoch
  if params.resume != '':
    resume_file = get_resume_file('%s/checkpoints/%s'%(params.save_dir, params.resume), params.resume_epoch)
    if resume_file is not None:
      start_epoch = model.resume(resume_file)
      print('  resume the training with at {} epoch (model file {})'.format(start_epoch, params.resume))
    else:
      raise ValueError('No resume file')
  # load pre-trained feature encoder
  else:
    if params.warmup == 'gg3b0':
      raise Exception('Must provide pre-trained feature-encoder file using --warmup option!')
    model.model.feature.load_state_dict(load_warmup_state('%s/checkpoints/%s'%(params.save_dir, params.warmup), params.method), strict=False)

  # training
  print('\n--- start the training ---')
  train(base_datamgr, datasets, aux_iter, val_loader, model, start_epoch, stop_epoch, params)
