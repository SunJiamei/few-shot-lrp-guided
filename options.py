import numpy as np
import os
import glob
import torch
import argparse

def parse_args(script):
  parser = argparse.ArgumentParser(description= 'few-shot script %s' %(script))
  parser.add_argument('--dataset', default='multi', help='miniImagenet/cub/cars/places/plantae, specify multi for training with multiple domains')
  parser.add_argument('--testset', default='cub', help='cub/cars/places/plantae, valid only when dataset=multi')
  parser.add_argument('--model', default='ResNet10', help='model: Conv{4|6} / ResNet{10|18|34}') # we use ResNet10 in the paper
  parser.add_argument('--method', default='relationnet',   help='baseline/baseline++/protonet/matchingnet/relationnet{_softmax}/gnnnet/relationnetlrp/gnnnetlrp/')
  parser.add_argument('--train_n_way' , default=5, type=int,  help='class num to classify for training')
  parser.add_argument('--test_n_way'  , default=5, type=int,  help='class num to classify for testing (validation) ')
  parser.add_argument('--n_shot'      , default=5, type=int,  help='number of labeled data in each class, same as n_support')
  parser.add_argument('--train_aug'   , action='store_true',  help='perform data augmentation or not during training ')
  parser.add_argument('--name'        , default='gnn', type=str, help='')
  parser.add_argument('--save_dir'    , default='./output', type=str, help='')
  parser.add_argument('--data_dir'    , default='../dataset', type=str, help='')
  parser.add_argument('--feature_wise_type', default=None, type=str,
                      help='[LFT, FT, LFTAdaIN] FT  will implement the feature wise transformation, LFT will implement learn to learn, adain will add the adaptive instance normalization')

  if script == 'train':
    parser.add_argument('--num_classes' , default=200, type=int, help='total number of classes in softmax, only used in baseline')
    parser.add_argument('--save_freq'   , default=25, type=int, help='Save frequency')
    parser.add_argument('--start_epoch' , default=0, type=int,help ='Starting epoch')
    parser.add_argument('--stop_epoch'  , default=400, type=int, help ='Stopping epoch')
    parser.add_argument('--resume'      , default='', type=str, help='continue from previous trained model with largest epoch')
    parser.add_argument('--resume_epoch', default=-1, type=int, help='')
    parser.add_argument('--warmup'      , default='baseline', type=str, help='continue from baseline, neglected if resume is true')
    parser.add_argument('--lr', default=0.001, type=float, help='the initial learning rate of the optimizer')
    parser.add_argument('--LUT_lr', default=[(200, 0.001), (300, 0.0003), (450, 0.0001)],
                        help="multistep to decay learning rate")
    parser.add_argument('--adj_lr', default=False, type=bool, help='whether to use the step learning rate')
  elif script == 'test':
    parser.add_argument('--transductive', default=False, help='whether use simple transductive evaluation')
    parser.add_argument('--split'       , default='novel', help='base/val/novel')
    parser.add_argument('--save_epoch', default=-1, type=int,help ='load the model trained in x epoch, use the best model if x is -1')
  else:
    raise ValueError('Unknown script')

  return parser.parse_args()

def get_assigned_file(checkpoint_dir,num):
  assign_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(num))
  return assign_file

def get_resume_file(checkpoint_dir, resume_epoch=-1):
  filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
  if len(filelist) == 0:
    return None

  filelist =  [ x  for x in filelist if os.path.basename(x) != 'best_model.tar' ]
  epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
  max_epoch = np.max(epochs)
  epoch = max_epoch if resume_epoch == -1 else resume_epoch
  resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
  return resume_file

def get_best_file(checkpoint_dir):
  best_file = os.path.join(checkpoint_dir, 'best_model.tar')
  if os.path.isfile(best_file):

    return best_file
  else:
    return get_resume_file(checkpoint_dir)

def load_warmup_state(filename, method):
  print('  load pre-trained model file: {}'.format(filename))
  warmup_resume_file = get_resume_file(filename)
  tmp = torch.load(warmup_resume_file)
  if tmp is not None:
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
      if 'relationnet' in method and "feature." in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      elif 'gnnnet' in method and 'feature.' in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      elif method == 'matchingnet' and 'feature.' in key and '.7.' not in key:
        newkey = key.replace("feature.","")
        state[newkey] = state.pop(key)
      else:
        state.pop(key)
  else:
    raise ValueError(' No pre-trained encoder file found!')
  return state

