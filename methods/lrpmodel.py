# This code is modified from https://github.com/floodsung/LearningToCompare_FSL
import os
from methods import backbone
import torch
import options
from LRPtools import lrp_wrapper
from LRPtools import lrp_presets
from methods.protonet import ProtoNet
from methods.matchingnet import MatchingNet
from methods.gnnnet import GnnNet
from methods.relationnet import RelationNet
from data.datamgr import SimpleDataManager, SetDataManager
from options import get_best_file, get_assigned_file
import LRPtools.utils as LRPutil
import copy
import numpy as np
import json
from PIL import Image
import torch.nn.functional as F
import torch.nn as nn
import yaml


def _get_sign_stabilizer(z, eps):
  sign_z = np.ones(z.shape)
  sign_z[z < 0] = -1
  return z + sign_z * eps

def lrp_backpropagate_linear(relevance_output, feature_input, weight, bias=None, ignore_bias=True):
  if len(weight.size())==2:
    V = weight.clone().detach()
    input_ = feature_input.clone().detach()
    # print('lrp', input_.shape, V.shape)
    Z = torch.mm(input_.clone().detach(), V.t())
    # print('lrp',Z.shape)
    if ignore_bias:
      # TODO this seems to be not done in iNNvestigate if biases are not ignored.
      assert bias == None
      Z += LRPutil.EPSILON * Z.sign()  # Z.sign() returns -1 or 0 or 1
      Z.masked_fill_(Z == 0, LRPutil.EPSILON)
    if not ignore_bias:
      assert bias!=None
      Z += bias.clone().detach()
    S = relevance_output.clone().detach() / Z
    C = torch.mm(S, V)
    # print('lrp', C.shape)
    R = input_ * C
    return R
  elif len(weight.size())==3:
    # print(relevance_output.shape, feature_input.shape, weight.shape)
    bs = relevance_output.size(0)
    relevance_input = []
    for i in range(bs):
      V = weight[i].clone().detach() #(J*N, N)
      input_ = feature_input[i].clone().detach() #(N, num_feature)
      relevance_outputi = relevance_output[i]
      # print(input_.shape, V.shape)
      Z = torch.mm(V, input_) #J*N, num_feature
      Z += LRPutil.EPSILON * Z.sign()  # Z.sign() returns -1 or 0 or 1
      Z.masked_fill_(Z == 0, LRPutil.EPSILON)
      S = relevance_outputi.clone().detach() / Z
      # print(S.shape)
      C = torch.mm(V.t(), S)
      R = input_ * C
      relevance_input.append(R.unsqueeze(0))
    return torch.cat(relevance_input, 0)


def explain_Gconv(relevance_output, Gconvlayer, Wi, feature_input):
  # one forward pass
  # print('fea_in',feature_input.shape)  # (bs, N, num_features)
  W_size = Wi.size()
  N = W_size[-2]
  J = W_size[-1]
  bs = W_size[0]
  W = Wi.split(1, 3) #[tensors, each with bs N, N, 1]
  W = torch.cat(W, 1).squeeze(3) # W is now a tensor of size (bs, J*N, N)
  output = torch.bmm(W, feature_input) # output has size (bs, J*N, num_features)
  num_feature = output.size(-1)
  output = output.split(N, 1)
  output = torch.cat(output, 2) # output has size (bs, N, J*num_features)
  output = output.view(-1, Gconvlayer.num_inputs)
  relevance_output = relevance_output.view(-1, Gconvlayer.num_outputs)
  relevance_output = lrp_backpropagate_linear(relevance_output, output, Gconvlayer.fc.weight)
  relevance_output = relevance_output.view(bs, N, J*num_feature)
  relevance_output = relevance_output.split(num_feature, -1)
  relevance_output = torch.cat(relevance_output, 1)
  relevance_feature_input = lrp_backpropagate_linear(relevance_output, feature_input, W)
  return relevance_feature_input

def project(x):
    absmax = np.max(np.abs(x))
    x = 1.0 * x / absmax
    if np.sum(x < 0):
        x = (x + 1) / 2
    else:
        x = x
    return x * 225

def explain_relationnet():
  # print(sys.path)
  params = options.parse_args('test')
  feature_model = backbone.model_dict['ResNet10']
  params.method = 'relationnet'
  params.dataset = 'miniImagenet'  # name relationnet --testset miniImagenet
  params.name = 'relationnet'
  params.testset = 'miniImagenet'
  params.data_dir = '/home/sunjiamei/work/fewshotlearning/dataset/'
  params.save_dir = '/home/sunjiamei/work/fewshotlearning/CrossDomainFewShot-master/output'

  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224
  split = params.split
  n_query = 1
  loadfile = os.path.join(params.data_dir, params.testset, split + '.json')
  few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
  data_datamgr = SetDataManager(image_size, n_query=n_query, **few_shot_params)
  data_loader = data_datamgr.get_data_loader(loadfile, aug=False)

  acc_all = []
  iter_num = 1000

  # model
  print('  build metric-based model')
  if params.method == 'protonet':
    model = ProtoNet(backbone.model_dict[params.model], **few_shot_params)
  elif params.method == 'matchingnet':
    model = MatchingNet(backbone.model_dict[params.model], **few_shot_params)
  elif params.method == 'gnnnet':
    model = GnnNet(backbone.model_dict[params.model], **few_shot_params)
  elif params.method in ['relationnet', 'relationnet_softmax']:
    if params.model == 'Conv4':
      feature_model = backbone.Conv4NP
    elif params.model == 'Conv6':
      feature_model = backbone.Conv6NP
    else:
      feature_model = backbone.model_dict[params.model]
    loss_type = 'LRPmse'
    model = RelationNet(feature_model, loss_type=loss_type, **few_shot_params)
  else:
    raise ValueError('Unknown method')

  checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.name)
  # print(checkpoint_dir)
  if params.save_epoch != -1:
    modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
  else:
    modelfile = get_best_file(checkpoint_dir)
    # print(modelfile)
  if modelfile is not None:
    tmp = torch.load(modelfile)
    try:
      model.load_state_dict(tmp['state'])
    except RuntimeError:
      print('warning! RuntimeError when load_state_dict()!')
      model.load_state_dict(tmp['state'], strict=False)
    except KeyError:
      for k in tmp['model_state']:  ##### revise latter
        if 'running' in k:
          tmp['model_state'][k] = tmp['model_state'][k].squeeze()
      model.load_state_dict(tmp['model_state'], strict=False)
    except:
      raise

  model = model.cuda()
  model.eval()
  model.n_query = n_query
  # ---test the accuracy on the test set to verify the model is loaded----
  acc = 0
  count = 0
  # for i, (x, y) in enumerate(data_loader):
  #   scores = model.set_forward(x)
  #   pred = scores.data.cpu().numpy().argmax(axis=1)
  #   y = np.repeat(range(model.n_way), n_query)
  #   acc += np.sum(pred == y)
  #   count += len(y)
  #   # print(1.0*acc/count)
  # print(1.0*acc/count)
  preset = lrp_presets.SequentialPresetA()

  feature_model = copy.deepcopy(model.feature)
  lrp_wrapper.add_lrp(feature_model, preset=preset)
  relation_model = copy.deepcopy(model.relation_module)
  # print(relation_model)
  lrp_wrapper.add_lrp(relation_model, preset=preset)
  with open('/home/sunjiamei/work/fewshotlearning/dataset/miniImagenet/class_to_readablelabel.json', 'r') as f:
    class_to_readable = json.load(f)
  explanation_save_dir = os.path.join(params.save_dir, 'explanations', params.name)
  if not os.path.isdir(explanation_save_dir):
    os.makedirs(explanation_save_dir)
  for i, (x, y, p) in enumerate(data_loader):
    '''x is the images with shape as n_way, n_support + n_querry, 3, img_size, img_size
       y is the global labels of the images with shape as (n_way, n_support + n_query)
       p is the image path as a list of tuples, length is n_query+n_support,  each tuple element is with length n_way'''
    if i >= 3:
      break
    label_to_readableclass, query_img_path, query_gt_class = LRPutil.get_class_label(p, class_to_readable,
                                                                                     model.n_query)
    z_support, z_query = model.parse_feature(x, is_feature=False)
    z_support = z_support.contiguous()
    z_proto = z_support.view(model.n_way, model.n_support, *model.feat_dim).mean(1)
    # print(z_proto.shape)
    z_query = z_query.contiguous().view(model.n_way * model.n_query, *model.feat_dim)
    # print(z_query.shape)
    # get relations with metric function
    z_proto_ext = z_proto.unsqueeze(0).repeat(model.n_query * model.n_way, 1, 1, 1, 1)
    # print(z_proto_ext.shape)
    z_query_ext = z_query.unsqueeze(0).repeat(model.n_way, 1, 1, 1, 1)

    z_query_ext = torch.transpose(z_query_ext, 0, 1)
    # print(z_query_ext.shape)
    extend_final_feat_dim = model.feat_dim.copy()
    extend_final_feat_dim[0] *= 2
    relation_pairs = torch.cat((z_proto_ext, z_query_ext), 2).view(-1, *extend_final_feat_dim)
    # print(relation_pairs.shape)
    relations = relation_model(relation_pairs)
    # print(relations)
    scores = relations.view(-1, model.n_way)
    preds = scores.data.cpu().numpy().argmax(axis=1)
    # print(preds.shape)
    relations = relations.view(-1, model.n_way)
    # print(relations)
    relations_sf = torch.softmax(relations, dim=-1)
    # print(relations_sf)
    relations_logits = torch.log(LRPutil.LOGIT_BETA * relations_sf / (1 - relations_sf))
    # print(relations_logits)
    # print(preds)
    relations_logits = relations_logits.view(-1, 1)
    relevance_relations = relation_model.compute_lrp(relation_pairs, target=relations_logits)
    # print(relevance_relations.shape)
    # print(model.feat_dim)
    relevance_z_query = torch.narrow(relevance_relations, 1, model.feat_dim[0], model.feat_dim[0])
    # print(relevance_z_query.shape)
    relevance_z_query = relevance_z_query.view(model.n_query * model.n_way, model.n_way,
                                               *relevance_z_query.size()[1:])
    # print(relevance_z_query.shape)
    query_img = x.narrow(1, model.n_support, model.n_query).view(model.n_way * model.n_query, *x.size()[2:])
    # query_img_copy = query_img.view(model.n_way, model.n_query, *x.size()[2:])
    # print(query_img.shape)
    for k in range(model.n_way):
      relevance_querry_cls = torch.narrow(relevance_z_query, 1, k, 1).squeeze(1)
      # print(relevance_querry_cls.shape)
      relevance_querry_img = feature_model.compute_lrp(query_img.cuda(), target=relevance_querry_cls)
      # print(relevance_querry_img.max(), relevance_querry_img.min())
      # print(relevance_querry_img.shape)
      for j in range(model.n_query * model.n_way):
        predict_class = label_to_readableclass[preds[j]]
        true_class = query_gt_class[int(j % model.n_way)][int(j // model.n_way)]
        explain_class = label_to_readableclass[k]
        img_name = query_img_path[int(j % model.n_way)][int(j // model.n_way)].split('/')[-1]
        if not os.path.isdir(os.path.join(explanation_save_dir, 'episode' + str(i), img_name.strip('.jpg'))):
          os.makedirs(os.path.join(explanation_save_dir, 'episode' + str(i), img_name.strip('.jpg')))
        save_path = os.path.join(explanation_save_dir, 'episode' + str(i), img_name.strip('.jpg'))
        if not os.path.exists(os.path.join(save_path, true_class + '_' + predict_class + img_name)):
          original_img = Image.fromarray(
            np.uint8(project(query_img[j].permute(1, 2, 0).cpu().numpy())))
          original_img.save(
            os.path.join(save_path, true_class + '_' + predict_class + img_name))

        img_relevance = relevance_querry_img.narrow(0, j, 1)
        print(predict_class, true_class, explain_class)
        # assert relevance_querry_cls[j].sum() != 0
        # assert img_relevance.sum()!=0
        hm = img_relevance.permute(0, 2, 3, 1).cpu().detach().numpy()
        hm = LRPutil.gamma(hm)
        hm = LRPutil.heatmap(hm)[0]
        hm = project(hm)
        hp_img = Image.fromarray(np.uint8(hm))
        hp_img.save(os.path.join(save_path,
                                 true_class + '_' + explain_class + '_lrp_hm.jpg'))
    # break

def explain_gnnnet():
  params = options.parse_args('test')
  feature_model = backbone.model_dict['ResNet10']
  params.method = 'gnnnet'
  params.dataset = 'miniImagenet'  # name relationnet --testset miniImagenet
  params.name = 'gnn'
  params.testset = 'miniImagenet'
  params.data_dir = '/home/sunjiamei/work/fewshotlearning/dataset/'
  params.save_dir = '/home/sunjiamei/work/fewshotlearning/CrossDomainFewShot-master/output'

  if 'Conv' in params.model:
    image_size = 84
  else:
    image_size = 224
  split = params.split
  n_query = 1
  loadfile = os.path.join(params.data_dir, params.testset, split + '.json')
  few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)
  data_datamgr = SetDataManager(image_size, n_query=n_query, **few_shot_params)
  data_loader = data_datamgr.get_data_loader(loadfile, aug=False)

  # model
  print('  build metric-based model')
  if params.method == 'protonet':
    model = ProtoNet(backbone.model_dict[params.model], **few_shot_params)
  elif params.method == 'matchingnet':
    model = MatchingNet(backbone.model_dict[params.model], **few_shot_params)
  elif params.method == 'gnnnet':
    model = GnnNet(backbone.model_dict[params.model], **few_shot_params)
  elif params.method in ['relationnet', 'relationnet_softmax']:
    if params.model == 'Conv4':
      feature_model = backbone.Conv4NP
    elif params.model == 'Conv6':
      feature_model = backbone.Conv6NP
    else:
      feature_model = backbone.model_dict[params.model]
    loss_type = 'LRP'
    model = RelationNet(feature_model, loss_type=loss_type, **few_shot_params)
  else:
    raise ValueError('Unknown method')

  checkpoint_dir = '%s/checkpoints/%s' % (params.save_dir, params.name)
  # print(checkpoint_dir)
  if params.save_epoch != -1:
    modelfile = get_assigned_file(checkpoint_dir, params.save_epoch)
  else:
    modelfile = get_best_file(checkpoint_dir)
    # print(modelfile)
  if modelfile is not None:
    tmp = torch.load(modelfile)
    try:
      model.load_state_dict(tmp['state'])
      print('loaded pretrained model')
    except RuntimeError:
      print('warning! RuntimeError when load_state_dict()!')
      model.load_state_dict(tmp['state'], strict=False)
    except KeyError:
      for k in tmp['model_state']:  ##### revise latter
        if 'running' in k:
          tmp['model_state'][k] = tmp['model_state'][k].squeeze()
      model.load_state_dict(tmp['model_state'], strict=False)
    except:
      raise

  model = model.cuda()
  model.eval()
  model.n_query = n_query
  # for module in model.modules():
  #   print(type(module))
  lrp_preset = lrp_presets.SequentialPresetA()
  feature_model = model.feature
  fc_encoder = model.fc
  gnn_net = model.gnn
  lrp_wrapper.add_lrp(fc_encoder,lrp_preset)
  # lrp_wrapper.add_lrp(feature_model, lrp_preset)
  # lrp_wrapper.add_lrp(fc_encoder,lrp_preset)
  # lrp_wrapper.add_lrp(feature_model, lrp_preset)



  # acc = 0
  # count = 0
  # tested the forward pass is correct by observing the accuracy
  # for i, (x, _, _) in enumerate(data_loader):
  #   x = x.cuda()
  #   support_label = torch.from_numpy(np.repeat(range(model.n_way), model.n_support)).unsqueeze(1)
  #   support_label = torch.zeros(model.n_way*model.n_support, model.n_way).scatter(1, support_label, 1).view(model.n_way, model.n_support, model.n_way)
  #   support_label = torch.cat([support_label, torch.zeros(model.n_way, 1, model.n_way)], dim=1)
  #   support_label = support_label.view(1, -1, model.n_way)
  #   support_label = support_label.cuda()
  #   x = x.view(-1, *x.size()[2:])
  #
  #   x_feature = feature_model(x)
  #   x_fc_encoded = fc_encoder(x_feature)
  #   z = x_fc_encoded.view(model.n_way, -1, x_fc_encoded.size(1))
  #   gnn_feature = [
  #     torch.cat([z[:, :model.n_support], z[:, model.n_support + i:model.n_support + i + 1]], dim=1).view(1, -1, z.size(2))
  #     for i in range(model.n_query)]
  #   gnn_nodes = torch.cat([torch.cat([z, support_label], dim=2) for z in gnn_feature], dim=0)
  #   scores = gnn_net(gnn_nodes)
  #   scores = scores.view(model.n_query, model.n_way, model.n_support + 1, model.n_way)[:, :, -1].permute(1, 0,
  #                                                                                                    2).contiguous().view(
  #     -1, model.n_way)
  #   pred = scores.data.cpu().numpy().argmax(axis=1)
  #   y = np.repeat(range(model.n_way), n_query)
  #   acc += np.sum(pred == y)
  #   count += len(y)
  #   # print(1.0*acc/count)
  # print(1.0*acc/count)
  with open('/home/sunjiamei/work/fewshotlearning/dataset/miniImagenet/class_to_readablelabel.json', 'r') as f:
    class_to_readable = json.load(f)
  explanation_save_dir = os.path.join(params.save_dir, 'explanations', params.name)
  if not os.path.isdir(explanation_save_dir):
    os.makedirs(explanation_save_dir)
  for batch_idx, (x, y, p) in enumerate(data_loader):
    print(p)
    label_to_readableclass, query_img_path, query_gt_class = LRPutil.get_class_label(p, class_to_readable,
                                                                                     model.n_query)
    x = x.cuda()
    support_label = torch.from_numpy(np.repeat(range(model.n_way), model.n_support)).unsqueeze(1) #torch.Size([25, 1])
    support_label = torch.zeros(model.n_way*model.n_support, model.n_way).scatter(1, support_label, 1).view(model.n_way, model.n_support, model.n_way)
    support_label = torch.cat([support_label, torch.zeros(model.n_way, 1, model.n_way)], dim=1)
    support_label = support_label.view(1, -1, model.n_way)
    support_label = support_label.cuda()  #torch.Size([1, 30, 5])
    x = x.contiguous()
    x = x.view(-1, *x.size()[2:])  #torch.Size([30, 3, 224, 224])
    x_feature = feature_model(x)  #torch.Size([30, 512])
    x_fc_encoded = fc_encoder(x_feature)  #torch.Size([30, 128])
    z = x_fc_encoded.view(model.n_way, -1, x_fc_encoded.size(1)) # (5,6,128)
    gnn_feature = [
      torch.cat([z[:, :model.n_support], z[:, model.n_support + i:model.n_support + i + 1]], dim=1).view(1, -1, z.size(2))
      for i in range(model.n_query)] # model.n_query is the number of query images for each class
    # gnn_feature is grouped into n_query groups. each group contains the support image features concatenated with one query image features.
    # print(len(gnn_feature), gnn_feature[0].shape)
    gnn_nodes = torch.cat([torch.cat([z, support_label], dim=2) for z in gnn_feature], dim=0)   # the features are concatenated with the one hot label. for the unknow image the one hot label is all zero


    #  perform gnn_net step by step
    #  the first iteration
    print('x', gnn_nodes.shape)
    W_init = torch.eye(gnn_nodes.size(1), device=gnn_nodes.device).unsqueeze(0).repeat(gnn_nodes.size(0), 1, 1).unsqueeze(
      3)  # (n_querry, n_way*(num_support + 1), n_way*(num_support + 1), 1)
    # print(W_init.shape)

    W1 = gnn_net._modules['layer_w{}'.format(0)](gnn_nodes,
                                              W_init)  # (n_querry, n_way*(num_support + 1), n_way*(num_support + 1), 2)
    # print(Wi.shape)
    x_new1 = F.leaky_relu(
      gnn_net._modules['layer_l{}'.format(0)]([W1, gnn_nodes])[1])  # (num_querry, num_support + 1, num_outputs)
    # print(x_new1.shape)  #torch.Size([1, 30, 48])
    gnn_nodes_1 = torch.cat([gnn_nodes, x_new1], 2)  # (concat more features)
    # print('gn1',gnn_nodes_1.shape) #torch.Size([1, 30, 181])

    #  the second iteration
    W2 = gnn_net._modules['layer_w{}'.format(1)](gnn_nodes_1,
                                              W_init)  # (n_querry, n_way*(num_support + 1), n_way*(num_support + 1), 2)
    x_new2 = F.leaky_relu(
      gnn_net._modules['layer_l{}'.format(1)]([W2, gnn_nodes_1])[1])  # (num_querry, num_support + 1, num_outputs)
    # print(x_new2.shape)
    gnn_nodes_2 = torch.cat([gnn_nodes_1, x_new2], 2)  # (concat more features)
    # print('gn2', gnn_nodes_2.shape)  #torch.Size([1, 30, 229])

    Wl = gnn_net.w_comp_last(gnn_nodes_2, W_init)
    # print(Wl.shape)  #torch.Size([1, 30, 30, 2])
    scores = gnn_net.layer_last([Wl, gnn_nodes_2])[1]  # (num_querry, num_support + 1, num_way)
    print(scores.shape)

    scores_sf = torch.softmax(scores, dim=-1)
    # print(scores_sf)

    gnn_logits = torch.log(LRPutil.LOGIT_BETA * scores_sf / (1 - scores_sf))
    gnn_logits = gnn_logits.view(-1, model.n_way, model.n_support+n_query, model.n_way)
    # print(gnn_logits)
    query_scores = scores.view(model.n_query, model.n_way, model.n_support + 1, model.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, model.n_way)
    preds = query_scores.data.cpu().numpy().argmax(axis=-1)
    # print(preds.shape)
    for k in range(model.n_way):
      mask = torch.zeros(5).cuda()
      mask[k] = 1
      gnn_logits_cls = gnn_logits.clone()
      gnn_logits_cls[:, :, -1] =  gnn_logits_cls[:, :, -1] * mask
      # print(gnn_logits_cls)
      # print(gnn_logits_cls.shape)
      gnn_logits_cls = gnn_logits_cls.view(-1, model.n_way)
      relevance_gnn_nodes_2 = explain_Gconv(gnn_logits_cls, gnn_net.layer_last, Wl, gnn_nodes_2)
      relevance_x_new2 = relevance_gnn_nodes_2.narrow(-1, 181,48)
      # relevance_gnn_nodes = relevance_gnn_nodes_2
      relevance_gnn_nodes_1 = explain_Gconv(relevance_x_new2, gnn_net._modules['layer_l{}'.format(1)], W2, gnn_nodes_1)
      relevance_x_new1 = relevance_gnn_nodes_1.narrow(-1, 133, 48)
      relevance_gnn_nodes = explain_Gconv(relevance_x_new1, gnn_net._modules['layer_l{}'.format(0)], W1, gnn_nodes)
      relevance_gnn_features = relevance_gnn_nodes.narrow(-1, 0, 128)
      print(relevance_gnn_features.shape)
      relevance_gnn_features += relevance_gnn_nodes_1.narrow(-1, 0, 128)
      relevance_gnn_features += relevance_gnn_nodes_2.narrow(-1, 0, 128)  #[2, 30, 128]
      relevance_gnn_features = relevance_gnn_features.view(n_query, model.n_way, model.n_support + 1, 128)
      for i in range(n_query):
        query_i = relevance_gnn_features[i][:, model.n_support:model.n_support+1]
        if i == 0:
          relevance_z = query_i
        else:
          relevance_z = torch.cat((relevance_z, query_i), 1)
      relevance_z = relevance_z.view(-1, 128)
      query_feature = x_feature.view(model.n_way, -1, 512)[:, model.n_support:]
      # print(query_feature.shape)
      query_feature = query_feature.contiguous()
      query_feature = query_feature.view(n_query*model.n_way, 512)
      # print(query_feature.shape)
      relevance_query_features = fc_encoder.compute_lrp(query_feature, target=relevance_z)
      # print(relevance_query_features.shape)
      # print(relevance_gnn_features.shape)
      # explain the fc layer and the image encoder
      query_images = x.view(model.n_way, -1, *x.size()[1:])[:, model.n_support:]
      query_images = query_images.contiguous()

      query_images = query_images.view(-1, *x.size()[1:]).detach()
      # print(query_images.shape)
      lrp_wrapper.add_lrp(feature_model, lrp_preset)
      relevance_query_images = feature_model.compute_lrp(query_images, target=relevance_query_features)
      print(relevance_query_images.shape)

      for j in range(n_query * model.n_way):
        predict_class = label_to_readableclass[preds[j]]
        true_class = query_gt_class[int(j % model.n_way)][int(j // model.n_way)]
        explain_class = label_to_readableclass[k]
        img_name = query_img_path[int(j % model.n_way)][int(j // model.n_way)].split('/')[-1]
        if not os.path.isdir(os.path.join(explanation_save_dir, 'episode' + str(batch_idx), img_name.strip('.jpg'))):
          os.makedirs(os.path.join(explanation_save_dir, 'episode' + str(batch_idx), img_name.strip('.jpg')))
        save_path = os.path.join(explanation_save_dir, 'episode' + str(batch_idx), img_name.strip('.jpg'))
        if not os.path.exists(os.path.join(save_path, true_class + '_' + predict_class + img_name)):
          original_img = Image.fromarray(
            np.uint8(project(query_images[j].permute(1, 2, 0).detach().cpu().numpy())))
          original_img.save(
            os.path.join(save_path, true_class + '_' + predict_class + img_name))

        img_relevance = relevance_query_images.narrow(0, j, 1)
        print(predict_class, true_class, explain_class)
        # assert relevance_querry_cls[j].sum() != 0
        # assert img_relevance.sum()!=0
        hm = img_relevance.permute(0, 2, 3, 1).cpu().detach().numpy()
        hm = LRPutil.gamma(hm)
        hm = LRPutil.heatmap(hm)[0]
        hm = project(hm)
        hp_img = Image.fromarray(np.uint8(hm))
        hp_img.save(os.path.join(save_path,
                                 true_class + '_' + explain_class + '_lrp_hm.jpg'))

    break





if __name__ == '__main__':

  explain_gnnnet()
  # explain_relationnet()
