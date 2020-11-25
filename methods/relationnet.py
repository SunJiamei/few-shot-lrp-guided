# This code is modified from https://github.com/floodsung/LearningToCompare_FSL

from methods import backbone
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from methods.meta_template import MetaTemplate
import utils
from LRPtools import lrp_wrapper
from LRPtools import lrp_presets
from LRPtools import utils as LRPutil
import copy
import gc
import utils
class RelationNet(MetaTemplate):
  def __init__(self, model_func,  n_way, n_support, tf_path=None, loss_type = 'mse'):
    super(RelationNet, self).__init__(model_func,  n_way, n_support, flatten=False, tf_path=tf_path)

    # loss function
    self.loss_type = loss_type  #'softmax' or 'mse'
    if 'mse' in self.loss_type:
      self.loss_fn = nn.MSELoss()
    else:
      self.loss_fn = nn.CrossEntropyLoss()

    # metric function
    self.relation_module = RelationModule( self.feat_dim , 8, self.loss_type ) #relation net features are not pooled, so self.feat_dim is [dim, w, h]
    self.method = 'RelationNet'


  def set_forward(self,x, is_feature = False):

    # get features
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_proto     = z_support.view( self.n_way, self.n_support, *self.feat_dim ).mean(1)
    z_query     = z_query.contiguous().view( self.n_way* self.n_query, *self.feat_dim )

    # get relations with metric function
    z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)
    z_query_ext = z_query.unsqueeze(0).repeat( self.n_way,1,1,1,1)
    z_query_ext = torch.transpose(z_query_ext,0,1)
    extend_final_feat_dim = self.feat_dim.copy()
    extend_final_feat_dim[0] *= 2
    relation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
    relations = self.relation_module(relation_pairs).view(-1, self.n_way)
    return relations
  def set_forward_transductive(self,x, candicate_num=35, is_feature = False):
    assert is_feature == True
    # get features
    z_support, z_query  = self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_support = z_support.view(self.n_way * self.n_support, *self.feat_dim)

    z_support_label = torch.from_numpy(np.repeat(range(self.n_way ), self.n_support)).cuda()

    z_query = z_query.contiguous().view(self.n_way * self.n_query, *self.feat_dim)
    z_query_ext = z_query.unsqueeze(0).repeat(self.n_way, 1, 1, 1, 1)
    z_query_ext = torch.transpose(z_query_ext, 0, 1)
    for trans_iter in range(2):

      z_support_one_hot_label = utils.one_hot(z_support_label, self.n_way)  #(num_train, n_way)
      z_support_one_hot_label = z_support_one_hot_label.transpose(0,1).cuda()  #(n_way, num_train)

      z_proto = torch.mm(z_support_one_hot_label, z_support.view(z_support.size(0), -1))  # (n_way, fea_dim)
      z_proto = z_proto.div(z_support_one_hot_label.sum(dim=1, keepdim=True).expand_as(z_proto))
      z_proto = z_proto.view(self.n_way, *self.feat_dim)
      z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)
      extend_final_feat_dim = self.feat_dim.copy()
      extend_final_feat_dim[0] *= 2
      relation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
      relations = self.relation_module(relation_pairs).view(-1, self.n_way)
      probs, preds = torch.max(relations, 1)
      top_k, top_k_id = torch.topk(probs, candicate_num * (trans_iter + 1), largest=True, sorted=True)  # (bs, K)
      candicate_img_index = top_k_id.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(candicate_num * (
              trans_iter + 1), *self.feat_dim)
      candicate_img = torch.gather(z_query, index=candicate_img_index, dim=0)
      candicate_label = torch.gather(preds, dim=0, index=top_k_id)
      z_support = torch.cat((z_support, candicate_img),dim=0)
      z_support_label = torch.cat((z_support_label, candicate_label),dim=0)
    z_support_one_hot_label = utils.one_hot(z_support_label, self.n_way)  # (num_train, n_way)
    z_support_one_hot_label = z_support_one_hot_label.transpose(0, 1).cuda()  # (n_way, num_train)
    z_proto = torch.mm(z_support_one_hot_label, z_support.view(z_support.size(0), -1))  # (n_way, fea_dim)
    z_proto = z_proto.div(z_support_one_hot_label.sum(dim=1, keepdim=True).expand_as(z_proto))
    z_proto = z_proto.view(self.n_way, *self.feat_dim)
    z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query * self.n_way, 1, 1, 1, 1)
    extend_final_feat_dim = self.feat_dim.copy()
    extend_final_feat_dim[0] *= 2
    relation_pairs = torch.cat((z_proto_ext, z_query_ext), 2).view(-1, *extend_final_feat_dim)
    relations = self.relation_module(relation_pairs).view(-1, self.n_way)
    return relations

  def set_forward_loss(self, x, epoch=None):
    y_local = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))

    scores = self.set_forward(x)
    if self.loss_type == 'mse':
      y_oh = utils.one_hot(y_local, self.n_way)
      y_oh = y_oh.cuda()
      loss = self.loss_fn(scores, y_oh)
    else:
      y_local = y_local.cuda()
      loss = self.loss_fn(scores, y_local)
    return scores, loss

# --- Convolution block used in the relation module ---
class RelationConvBlock(nn.Module):
  maml = False
  maml_adain = False
  assert (maml and maml_adain) == False
  def __init__(self, indim, outdim, padding = 0):
    super(RelationConvBlock, self).__init__()
    self.indim  = indim
    self.outdim = outdim
    if self.maml or self.maml_adain:
      self.C      = backbone.Conv2d_fw(indim, outdim, 3, padding=padding)
      self.BN     = backbone.BatchNorm2d_fw(outdim, momentum=1, track_running_stats=False)
    else:
      self.C      = nn.Conv2d(indim, outdim, 3, padding = padding )
      self.BN     = nn.BatchNorm2d(outdim, momentum=1, affine=True, track_running_stats=False)
    self.relu   = nn.ReLU()
    self.pool   = nn.MaxPool2d(2)

    self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

    for layer in self.parametrized_layers:
      backbone.init_layer(layer)

    self.trunk = nn.Sequential(*self.parametrized_layers)

  def forward(self,x):
    out = self.trunk(x)
    # print('x', x.shape)
    # out = self.C(x)
    # print('weight.shape', self.C.weight.shape)
    # print(out.shape)
    # out = self.BN(out)
    # print(out.shape)
    # out = self.relu(out)
    # print(out.shape)
    # out = self.pool(out)
    # print(out.shape)
    return out

# --- Relation module adopted in RelationNet ---
class RelationModule(nn.Module):
  maml = False
  maml_adain = False
  assert (maml and maml_adain) == False
  def __init__(self,input_size,hidden_size, loss_type = 'mse'):
    super(RelationModule, self).__init__()

    self.loss_type = loss_type
    padding = 1 if ( input_size[1] <10 ) and ( input_size[2] <10 ) else 0 # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling
    # print(padding)
    self.layer1 = RelationConvBlock(input_size[0]*2, input_size[0], padding = padding )
    self.layer2 = RelationConvBlock(input_size[0], input_size[0], padding = padding )

    shrink_s = lambda s: int((int((s- 2 + 2*padding)/2)-2 + 2*padding)/2)

    if self.maml or self.maml_adain:
      self.fc1 = backbone.Linear_fw( input_size[0]* shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size )
      self.fc2 = backbone.Linear_fw( hidden_size,1)
    else:
      self.fc1 = nn.Linear( input_size[0]* shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size )
      self.fc2 = nn.Linear( hidden_size,1)

  def forward(self,x):
    out = self.layer1(x)
    # print(out.shape)
    out = self.layer2(out)
    # print(out.shape)
    out = out.view(out.size(0),-1)
    # print(out.shape)
    out = F.relu(self.fc1(out))
    # print(out.shape, self.fc1.weight.shape)
    # print(out.sum(dim=-1))
    if self.loss_type == 'mse':
      out = torch.sigmoid(self.fc2(out))
    elif self.loss_type == 'softmax':
      out = self.fc2(out)
    elif self.loss_type == 'LRPmse':
      out = self.fc2(out)
    # print(out.shape)
    return out


# --- Relationnet with LRP weighted features ---

class RelationNetLRP(RelationNet):
  def __init__(self, model_func,  n_way, n_support, tf_path=None, loss_type = 'mse'):
    super(RelationNetLRP, self).__init__(model_func,  n_way, n_support, tf_path=tf_path, loss_type=loss_type)
    self.preset = lrp_presets.SequentialPresetA()
    self.scale_cls = 20
    self.lrptemperature = 1
    self.method = 'RelationNetLPR'
    self.total_epoch = 200

  def get_feature_relevance(self, relation_pairs, relations):
    model = copy.deepcopy(self.relation_module)
    lrp_wrapper.add_lrp(model, preset=self.preset)
    relations_sf = torch.softmax(relations, dim=-1)
    assert not torch.isnan(relations_sf.sum())
    assert not torch.isinf(relations_sf.sum())
    relations_logits = torch.log(LRPutil.LOGIT_BETA * (relations_sf +LRPutil.EPSILON)/ (torch.tensor([1 + LRPutil.EPSILON]).cuda() - relations_sf))
    relations_logits = relations_logits.view(-1, 1)
    assert not torch.isnan(relations_logits.sum())
    assert not torch.isinf(relations_logits.sum())
    relevance_relations = model.compute_lrp(relation_pairs, target=relations_logits) # (n_way*n_querry * n_support, 2* feature_dim, f_h, f_w)
    assert not torch.isnan(relevance_relations.sum())
    assert not torch.isinf(relevance_relations.sum())
    '''normalize the prototype and the query separately '''
    relevance_prototype = relevance_relations.narrow(1, 0, self.feat_dim[0])
    relevance_query = relevance_relations.narrow(1, self.feat_dim[0], self.feat_dim[0])
    relevance_prototype = LRPutil.normalize_relevance(relevance_prototype, dim=1, temperature=1)
    relevance_query = LRPutil.normalize_relevance(relevance_query, dim=1, temperature=1)
    normalized_relevance = LRPutil.normalize_relevance(relevance_relations, dim=1, temperature=1)
    del model
    gc.collect()
    return normalized_relevance, relevance_prototype, relevance_query


  def set_forward_loss(self, x, epoch=None):
    # print(y.shape)
    y_local = torch.from_numpy(np.repeat(range( self.n_way ), self.n_query ))

    scores = self.set_forward(x)
    if isinstance(scores, tuple) and len(scores) == 2:
      relations, relations_lrp = scores
      y_oh = utils.one_hot(y_local, self.n_way)
      y_oh = y_oh.cuda()
      loss1 = self.loss_fn(relations, y_oh)
      loss2 = self.loss_fn(relations_lrp, y_oh)
      if self.n_support == 5:
        loss = loss1 + loss2
      if self.n_support == 1:
        loss = loss1 + 0.5 * loss2
      return relations, loss
    elif 'mse' in self.loss_type:
      y_oh = utils.one_hot(y_local, self.n_way)
      y_oh = y_oh.cuda()
      loss = self.loss_fn(scores, y_oh)
    else:
      y_local = y_local.cuda()
      loss = self.loss_fn(scores, y_local)
    return scores, loss

  def set_forward(self,x, is_feature = False):

    # get features
    z_support, z_query= self.parse_feature(x,is_feature)
    z_support   = z_support.contiguous()
    z_proto     = z_support.view( self.n_way, self.n_support, *self.feat_dim ).mean(1)
    z_query     = z_query.contiguous().view( self.n_way* self.n_query, *self.feat_dim )

    # get relations with metric function
    z_proto_ext = z_proto.unsqueeze(0).repeat(self.n_query* self.n_way,1,1,1,1)
    z_query_ext = z_query.unsqueeze(0).repeat( self.n_way,1,1,1,1)
    z_query_ext = torch.transpose(z_query_ext,0,1)   # n_querry * n_way, n_way, *fea_dim
    extend_final_feat_dim = self.feat_dim.copy()
    extend_final_feat_dim[0] *= 2
    relation_pairs = torch.cat((z_proto_ext,z_query_ext),2).view(-1, *extend_final_feat_dim)
    # print('relations', relations.sum())
    # print(relation_pairs.sum())
    relations = self.relation_module(relation_pairs).view(-1, self.n_way)
    if self.training:

      self.relation_module.eval()
      relation_pairs_lrp = relation_pairs.detach()
      relations_lrp = self.relation_module(relation_pairs_lrp).view(-1, self.n_way)
      relevance_relation_pairs, _,_ = self.get_feature_relevance(relation_pairs_lrp, relations_lrp)
      relation_pairs = relation_pairs * relevance_relation_pairs
      self.relation_module.train()
      relations_lrp  = self.relation_module(relation_pairs).view(-1, self.n_way)
      relations = torch.sigmoid(relations)
      relations_lrp = torch.sigmoid(relations_lrp)
      return relations, relations_lrp

    if 'mse' in self.loss_type:
      relations = torch.sigmoid(relations)
    return relations