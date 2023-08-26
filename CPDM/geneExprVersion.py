
import torch
import pandas as pd
import os
from PIL import Image
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

def prepareGeneDataset_end2end(SideImgPath,SideImgIDpath,geneExprPath,mutationStatusPath,clinicalDataPath,gene,cParam=None):
  # prepare images
  def getMatchedImg(matchedImg) -> list:
    result = []
    for dirpath, dirnames, filenames in os.walk(matchedImg):
        for filepath in filenames:
            result.append(os.path.join(dirpath, filepath))
    result.sort()
    return result
  dic = getMatchedImg(SideImgPath)
  tensors = []
  for pa in dic:
    img = Image.open(pa).convert('L')
    np_img = np.array(img) / 255.0
    image = np.asarray([np_img], dtype=np.float32)
    each = torch.from_numpy(image).to(device)
    tensors.append(each)

  # prepare gene expressions
  geneExprData = pd.read_csv(geneExprPath).transpose()
  geneExprData.columns = geneExprData.iloc[0]
  geneExprData = geneExprData[1:]
  geneExprData.columns.name = None
  geneExprData = geneExprData.reset_index()
  geneExprData = geneExprData.rename(columns={'index': 'ID'})

  geneExprDataList = []
  sideViewIDs = []
  for each in pd.read_csv(SideImgIDpath).values.tolist():
    id = each[1].split('_')[0].replace('-','.')
    sideViewIDs.append(id)
    label = geneExprData.loc[geneExprData['ID']==id]
    geneExprDataList.append(label.values[0])
  pairedGeneExprs = pd.DataFrame(geneExprDataList, columns=['ID','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15'])
  result = pairedGeneExprs.drop(['ID'],axis=1)
  result = result.to_numpy().astype(np.float32)
  for each in range(len(result)):
    temp = result[each]
    min_value = np.min(temp)
    max_value = np.max(temp)
    normalized = (temp - min_value) / (max_value - min_value)
    result[each] = normalized
  t1 = torch.from_numpy(result).to(device)
  t1 = t1[:,None]
  btfs = []
  for te in t1:
    btfs.append(te)

  mutationStatus = pd.read_csv(mutationStatusPath).drop(['Altered'],axis=1)
  realImgMutateStatus = pd.DataFrame(columns = ['ID', 'AKT1', 'ARID1A', 'BRCA1', 'CASP8', 'CBFB', 'CDH1', 'CDKN1B',
        'CHD4', 'CTCF', 'ERBB2', 'FBXW7', 'FOXA1', 'GATA3', 'GPS2', 'KMT2C',
        'KRAS', 'MAP2K4', 'MAP3K1', 'NCOR1', 'NF1', 'PIK3CA', 'PIK3R1', 'PTEN',
        'PTPRD', 'RB1', 'RUNX1', 'SF3B1', 'TBX3', 'TP53'])
  for each in sideViewIDs:
    temp = mutationStatus.loc[mutationStatus['ID']==each]
    realImgMutateStatus = realImgMutateStatus.append(temp, ignore_index=True)

  status = realImgMutateStatus[gene].tolist()

  #### clinical
  clinical = pd.read_csv(clinicalDataPath)
  realImgClinical = pd.DataFrame(columns = ['ID', 'er', 'pr', 'her2'])
  for each in sideViewIDs:
    temp = clinical.loc[clinical['ID']==each]
    realImgClinical = realImgClinical.append(temp, ignore_index=True)
  clinicalStatus = realImgClinical[cParam].tolist()
  ####

  btfs_train = []
  btfs_test = []
  tensors_train = []
  tensors_test = []
  status_train = []
  status_test = []
  clinicalStatus_train = []
  clinicalStatus_test = []

  # test = [1,5,7,12,16,19,24,33]
  test = [1,16,24,33]
  for te in range(len(tensors)):
    if te in test:
      btfs_test.append(btfs[te])
      tensors_test.append(tensors[te])
      status_test.append(status[te])
      clinicalStatus_test.append(clinicalStatus[te])
    else:
      btfs_train.append(btfs[te])
      tensors_train.append(tensors[te])
      status_train.append(status[te])
      clinicalStatus_train.append(clinicalStatus[te])

  return btfs_train, btfs_test, tensors_train, tensors_test, status_train, status_test, clinicalStatus_train, clinicalStatus_test




""" Diffusion - Cross Attention """

import torch, os, sys
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# condition_flag = True
class CrossAttention(nn.Module):
    def __init__(self, dim, feature_dim, heads=8, dim_head=64): # heads=8
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Conv2d(dim, inner_dim, kernel_size=1, stride=1, padding=0)
        self.to_kv = nn.Linear(feature_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, x, context):
        b, _, h, w = x.shape
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = rearrange(q, 'b (h d) i j -> b h (i j) d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * (1. / math.sqrt(k.shape[-1]))
        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h (i j) d -> b (h d) i j', h=self.heads, i=h, j=w)

        return self.to_out(out)


def prepareDataset(SideImgPath,SideBTFpath,allBTFpath):
  """ Maybe, this needs to be more general !!! """
  def getMatchedImg(matchedImg) -> list:
      result = []
      for dirpath, dirnames, filenames in os.walk(matchedImg):
          for filepath in filenames:
              result.append(os.path.join(dirpath, filepath))
      result.sort()
      return result
  dic = getMatchedImg(SideImgPath)
  tensors = []
  for pa in dic:
    img = Image.open(pa).convert('L')
    np_img = np.array(img) / 255.0
    image = np.asarray([np_img], dtype=np.float32)
    each = torch.from_numpy(image).to(device)
    tensors.append(each)

  # prepare BTFs
  sv = pd.read_csv(SideBTFpath)
  sv = sv.values.tolist()
  btf = []
  for sve in sv:
      btf.append(sve[1])

  df = pd.read_csv(allBTFpath,names=['ID','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17'])
  df_empty = pd.DataFrame(columns=['ID','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17'])
  for bt in btf:
      label = df.loc[df['ID']==bt]
      df_empty=df_empty.append(label,ignore_index=True)
  result = df_empty.drop(['ID'],axis=1)
  result = result.to_numpy().astype(np.float32)

  for each in range(len(result)):
    temp = result[each]
    min_value = np.min(temp)
    max_value = np.max(temp)
    normalized = (temp - min_value) / (max_value - min_value)
    result[each] = normalized

  t1 = torch.from_numpy(result).to(device)
  t1 = t1[:,None]

  btfs = []
  for te in t1:
      btfs.append(te)

  btfs_train = []
  btfs_test = []
  tensors_train = []
  tensors_test = []

  # test = [1,5,7,12,16,19,24,33]
  test = [1,16,24,33]
  for te in range(len(tensors)):
    if te in test:
      btfs_test.append(btfs[te])
      tensors_test.append(tensors[te])
    else:
      btfs_train.append(btfs[te])
      tensors_train.append(tensors[te])

  return btfs_train, btfs_test, tensors_train, tensors_test


def prepareDataset_end2end(SideImgPath,SideBTFpath,allBTFpath,mutationStatusPath,clinicalDataPath,gene,cParam=None):
  """ Maybe, this needs to be more general !!! """
  def getMatchedImg(matchedImg) -> list:
      result = []
      for dirpath, dirnames, filenames in os.walk(matchedImg):
          for filepath in filenames:
              result.append(os.path.join(dirpath, filepath))
      result.sort()
      return result
  dic = getMatchedImg(SideImgPath)
  tensors = []
  for pa in dic:
    img = Image.open(pa).convert('L')
    np_img = np.array(img) / 255.0
    image = np.asarray([np_img], dtype=np.float32)
    each = torch.from_numpy(image).to(device)
    tensors.append(each)

  # prepare BTFs
  sv = pd.read_csv(SideBTFpath)
  sv = sv.values.tolist()
  btf = []
  for sve in sv:
      btf.append(sve[1])

  df = pd.read_csv(allBTFpath,names=['ID','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17'])
  df_empty = pd.DataFrame(columns=['ID','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12','V13','V14','V15','V16','V17'])
  for bt in btf:
      label = df.loc[df['ID']==bt]
      df_empty=df_empty.append(label,ignore_index=True)
  result = df_empty.drop(['ID'],axis=1)
  result = result.to_numpy().astype(np.float32)

  for each in range(len(result)):
    temp = result[each]
    min_value = np.min(temp)
    max_value = np.max(temp)
    normalized = (temp - min_value) / (max_value - min_value)
    result[each] = normalized

  t1 = torch.from_numpy(result).to(device)
  t1 = t1[:,None]

  btfs = []
  for te in t1:
      btfs.append(te)

  sideViewBTFID = pd.read_csv(SideBTFpath)['1'].tolist()
  for each in range(len(sideViewBTFID)):
    sideViewBTFID[each] = sideViewBTFID[each].split('_')[0].replace('-','.')

  mutationStatus = pd.read_csv(mutationStatusPath).drop(['Altered'],axis=1)
  realImgMutateStatus = pd.DataFrame(columns = ['ID', 'AKT1', 'ARID1A', 'BRCA1', 'CASP8', 'CBFB', 'CDH1', 'CDKN1B',
        'CHD4', 'CTCF', 'ERBB2', 'FBXW7', 'FOXA1', 'GATA3', 'GPS2', 'KMT2C',
        'KRAS', 'MAP2K4', 'MAP3K1', 'NCOR1', 'NF1', 'PIK3CA', 'PIK3R1', 'PTEN',
        'PTPRD', 'RB1', 'RUNX1', 'SF3B1', 'TBX3', 'TP53'])
  for each in sideViewBTFID:
    temp = mutationStatus.loc[mutationStatus['ID']==each]
    realImgMutateStatus = realImgMutateStatus.append(temp, ignore_index=True)

  status = realImgMutateStatus[gene].tolist()

  #### clinical
  clinical = pd.read_csv(clinicalDataPath)
  realImgClinical = pd.DataFrame(columns = ['ID', 'er', 'pr', 'her2'])
  for each in sideViewBTFID:
    temp = clinical.loc[clinical['ID']==each]
    realImgClinical = realImgClinical.append(temp, ignore_index=True)
  clinicalStatus = realImgClinical[cParam].tolist()
  ####

  btfs_train = []
  btfs_test = []
  tensors_train = []
  tensors_test = []
  status_train = []
  status_test = []
  clinicalStatus_train = []
  clinicalStatus_test = []

  # test = [1,5,7,12,16,19,24,33]
  test = [1,16,24,33]
  for te in range(len(tensors)):
    if te in test:
      btfs_test.append(btfs[te])
      tensors_test.append(tensors[te])
      status_test.append(status[te])
      clinicalStatus_test.append(clinicalStatus[te])
    else:
      btfs_train.append(btfs[te])
      tensors_train.append(tensors[te])
      status_train.append(status[te])
      clinicalStatus_train.append(clinicalStatus[te])

  return btfs_train, btfs_test, tensors_train, tensors_test, status_train, status_test, clinicalStatus_train, clinicalStatus_test



from torch.utils.data import Dataset
class load_data(Dataset):
  def __init__(self, images=None, btfs=None, data_type="train", device='cuda', transform=None):
      self.images = images
      self.btfs = btfs
      self.data_type = data_type
      self.device = device
      self.transform = transform
  def __len__(self):
      if len(self.images) != len(self.btfs):
        print('Length Error')
        raise
      return len(self.images)
  def __getitem__(self, idx):
      image = self.images[idx]
      btf = self.btfs[idx]
      if self.transform!=None:
        image = self.transform(image)
      return image, btf


class load_data_end2end(Dataset):
  def __init__(self, images=None, btfs=None, status=None, data_type="train", device='cuda', transform=None):
      self.images = images
      self.btfs = btfs
      self.status = status
      self.data_type = data_type
      self.device = device
      self.transform = transform
  def __len__(self):
      if len(self.images) != len(self.btfs) or len(self.images) != len(self.status):
        print('Length Error')
        raise
      return len(self.images)
  def __getitem__(self, idx):
      image = self.images[idx]
      btf = self.btfs[idx]
      status = self.status[idx]
      if self.transform!=None:
        image = self.transform(image)
      return image, btf, status


def get_dataloaderM(batch_size, btfs_train, btfs_test, tensors_train, tensors_test):
  load_train_set = load_data(images=tensors_train, btfs=btfs_train)
  train_loader = DataLoader(load_train_set, batch_size=batch_size, shuffle=True)
  load_val_set = load_data(images=tensors_test, btfs=btfs_test, data_type="val")
  val_loader = DataLoader(load_val_set, batch_size=batch_size, shuffle=False)
  # return load_train_set, load_val_set, train_loader, val_loader
  return train_loader, val_loader


def get_dataloaderM_end2end(batch_size, btfs_train, btfs_test, tensors_train, tensors_test, status_train, status_test):
  load_train_set = load_data_end2end(images=tensors_train, btfs=btfs_train, status=status_train)
  train_loader = DataLoader(load_train_set, batch_size=batch_size, shuffle=True)
  load_val_set = load_data_end2end(images=tensors_test, btfs=btfs_test, status=status_test, data_type="val")
  val_loader = DataLoader(load_val_set, batch_size=batch_size, shuffle=False)
  # return load_train_set, load_val_set, train_loader, val_loader
  return train_loader, val_loader


import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch import nn, einsum
import torch.nn.functional as F

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )

def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class PreNormCrossAttention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.norm = nn.BatchNorm2d(dim)
        # self.norm = nn.InstanceNorm2d(dim)  # possible tuning
        self.fn = fn

    def forward(self, x, context):
        return self.fn(self.norm(x), context)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        # self_condition=False,
        resnet_block_groups=4,
        feature_vector_dim = 15
    ):
        super().__init__()
        # determine dimensions
        self.channels = channels
        self.imagSize = dim
        # self.self_condition = self_condition
        # input_channels = channels * (2 if self_condition else 1)

        # self.btf_embedding = torch.nn.Linear(17, dim*dim, bias=False)
        # self.btf_embedding.weight.requires_grad = False
        # self.btf_embedding = self.btf_embedding

        input_channels = 1  #????
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 1, padding=0)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4
        # self.cross_attn = Residual(PreNorm(dim, CrossAttention(dim))) #### added
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        # feature_vector_dim = 17
        self.mid_attn = Residual(PreNormCrossAttention(mid_dim, CrossAttention(mid_dim,feature_vector_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, feature_vector=None):

        # inner product
        # btf = feature_vector.float()
        # btf = self.btf_embedding(btf) #.double()
        # btf = btf.view(feature_vector.shape[0], 1, self.imagSize, self.imagSize)
        # ## x = x.view(x.shape[0], 1, 32, self.cube_len,self.cube_len)
        # x = x + x * btf
        #

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)
        feature_vector=feature_vector.view(x.size(0), 1, -1).repeat(1, x.size(2) * x.size(3), 1)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x, feature_vector)
        x = self.mid_block2(x, t)
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def getArg(timesteps):
  betas = linear_beta_schedule(timesteps=timesteps)
  alphas = 1. - betas
  alphas_cumprod = torch.cumprod(alphas, axis=0)
  alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

  sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
  sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
  sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
  posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

  return {'betas':betas,
          'sqrt_recip_alphas':sqrt_recip_alphas,
          'sqrt_alphas_cumprod':sqrt_alphas_cumprod,
          'sqrt_one_minus_alphas_cumprod':sqrt_one_minus_alphas_cumprod,
          'posterior_variance':posterior_variance
          }

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def q_sample(x_start, t, timesteps, noise=None):
    args = getArg(timesteps)
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(args['sqrt_alphas_cumprod'], t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        args['sqrt_one_minus_alphas_cumprod'], t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise



def p_losses(denoise_model, x_start, t, timesteps, condition=None, noise=None, loss_type="l1", cycle_weight=0.1):

    ##
    # btf = feature_vector.float()
    # btf = self.btf_embedding(btf) #.double()
    # btf = btf.view(feature_vector.shape[0], 1, self.imagSize, self.imagSize)
    # ## x = x.view(x.shape[0], 1, 32, self.cube_len,self.cube_len)
    # x = x + x * btf
    ##

    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, timesteps=timesteps, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, condition)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "hyber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    # Compute cycle consistency term
    # x_recon = p_sample(model=denoise_model, x=x_noisy, t=t, t_index=t.cpu().numpy()[0], condition=condition)
    # cycle_term = F.mse_loss(x_start, x_recon)
    # loss += cycle_weight * cycle_term

    # CNN Loss - end to end

    return loss



def p_losses_end2end(denoise_model, CNN_model, criterion, labels, x_start, t, timesteps=1500,
                     condition=None, noise=None, loss_type="l1", device='cuda', CNN_weight=0.1,
                     cycle_weight=0.1):
    #### put it in q_sample?
    # btf_embedding = torch.nn.Linear(17, x_start.shape[2]*x_start.shape[2], bias=False).to(device)
    # btf_embedding.weight.requires_grad = False

    # btf = condition.float()
    # btf = btf_embedding(btf) #.double()
    # btf = btf.view(condition.shape[0], 1, x_start.shape[2], x_start.shape[2])
    # x_start = x_start + x_start * btf
    ####

    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t, timesteps=timesteps, noise=noise)
    predicted_noise = denoise_model(x_noisy, t, condition)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "hyber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    t_list = t.cpu().numpy().tolist()
    if 0 in t_list:
      # print('0 - CNN')
      tIndex = t_list.index(0)
      x_recon = p_sample(model=denoise_model, x=x_noisy, t=t, t_index=t.cpu().numpy()[0], condition=condition, timesteps=timesteps) * 255
      outputs = CNN_model(x_recon[tIndex])
      CNN_loss = criterion(outputs.squeeze(), labels[tIndex].float())
      loss = loss + CNN_weight * CNN_loss

    return loss



@torch.no_grad()
def p_sample(model, x, t, t_index, condition, timesteps):
    args = getArg(timesteps)
    betas_t = extract(args['betas'], t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        args['sqrt_one_minus_alphas_cumprod'], t, x.shape
    )
    # sqrt_recip_alphas = ['sqrt_recip_alphas']
    sqrt_recip_alphas_t = extract(args['sqrt_recip_alphas'], t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t, condition) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(args['posterior_variance'], t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape, condition, timesteps):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)

    ###
    btf_embedding = torch.nn.Linear(15, shape[2]*shape[2], bias=False).to('cuda')
    btf_embedding.weight.requires_grad = False

    btf = condition.float()
    btf = btf_embedding(btf) #.double()
    btf = btf.view(condition.shape[0], 1, shape[2], shape[2])
    img = img + img * btf
    ###

    imgs = []
    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i, condition, timesteps)
        imgs.append(img.cpu().numpy())
    return imgs * 255

@torch.no_grad()
def sample(model, image_size, timesteps, batch_size=4, channels=3, condition=None):  # batch
    return p_sample_loop(model, (batch_size, channels, image_size, image_size), condition, timesteps)

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def train_pdm(config):
  model,optimizer,use_checkpoint = config['model'],config['optimizer'],config['use_checkpoint']
  checkpoint_path,epochs,train_loader = config['checkpoint_path'],config['epochs'],config['train_loader']
  device,loss_type,val_loader,timesteps = config['device'],config['loss_type'],config['val_loader'],config['timesteps']

  loss_history = {"train": [],"val": []}
  if os.path.isfile(checkpoint_path) and use_checkpoint:
      checkpoint = torch.load(checkpoint_path)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      start_epoch = checkpoint['epoch']
      loss_history['train'] = checkpoint['train_loss_history']
      loss_history['val'] = checkpoint['val_loss_history']
      timesteps = checkpoint['timesteps']
  else:
      start_epoch = 0

  for epoch in tqdm(range(start_epoch, epochs)):
      model.train()
      train_loss = 0
      for step, batch in enumerate(train_loader):
          optimizer.zero_grad()
          btf = batch[1].to(device)
          mri = batch[0].to(device)
          batch_size = btf.shape[0]
          t = torch.randint(0, timesteps, (batch_size,), device=device).long()
          loss = p_losses(model, mri, t, timesteps, condition=btf, loss_type=loss_type)
          train_loss += loss.item()
          loss.backward(retain_graph=True)
          optimizer.step()
      train_loss /= len(train_loader)
      loss_history['train'].append(train_loss)

      model.eval()
      with torch.no_grad():
          val_loss = 0
          for batch in val_loader:
              btf = batch[1].to(device)
              mri = batch[0].to(device)
              batch_size = btf.shape[0]
              t = torch.randint(0, timesteps, (batch_size,), device=device).long()
              loss = p_losses(model, mri, t, timesteps, condition=btf, loss_type=loss_type)
              val_loss += loss.item()
          val_loss /= len(val_loader)
          loss_history['val'].append(val_loss)

      torch.save({
                  'epoch': epoch + 1,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'train_loss_history': loss_history['train'],
                  'val_loss_history': loss_history['val'],
                  'timesteps': timesteps
              }, checkpoint_path)



def train_end2end(config):
  model,optimizer,use_checkpoint = config['model'],config['optimizer'],config['use_checkpoint']
  checkpoint_path,epochs,train_loader = config['checkpoint_path'],config['epochs'],config['train_loader']
  device,loss_type,val_loader,timesteps = config['device'],config['loss_type'],config['val_loader'],config['timesteps']
  CNNModel,criterion,CNN_weight = config['CNNModel'], config['criterion'],config['CNN_weight']

  loss_history = {"train": [],"val": []}
  if use_checkpoint and os.path.isfile(checkpoint_path):
      checkpoint = torch.load(checkpoint_path)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      start_epoch = checkpoint['epoch']
      loss_history['train'] = checkpoint['train_loss_history']
      loss_history['val'] = checkpoint['val_loss_history']
      timesteps = checkpoint['timesteps']
  else:
      start_epoch = 0

  for epoch in tqdm(range(start_epoch, epochs)):
      model.train()
      train_loss = 0
      for step, batch in enumerate(train_loader):
          optimizer.zero_grad()
          label = batch[2].to(device)
          btf = batch[1].to(device)
          mri = batch[0].to(device)
          batch_size = btf.shape[0]
          t = torch.randint(0, timesteps, (batch_size,), device=device).long()
          loss = p_losses_end2end(model, CNNModel, criterion, label, mri, t, timesteps=timesteps,
                                  condition=btf, loss_type=loss_type,CNN_weight=CNN_weight)
          train_loss += loss.item()
          loss.backward(retain_graph=True)
          optimizer.step()
      train_loss /= len(train_loader)
      loss_history['train'].append(train_loss)

      model.eval()
      with torch.no_grad():
          val_loss = 0
          for batch in val_loader:
              label = batch[2].to(device)
              btf = batch[1].to(device)
              mri = batch[0].to(device)
              batch_size = btf.shape[0]
              t = torch.randint(0, timesteps, (batch_size,), device=device).long()
              loss = p_losses_end2end(model, CNNModel, criterion, label, mri, t, timesteps=timesteps,
                                  condition=btf, loss_type=loss_type,CNN_weight=CNN_weight)
              val_loss += loss.item()
          val_loss /= len(val_loader)
          loss_history['val'].append(val_loss)

      torch.save({
                  'epoch': epoch + 1,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'train_loss_history': loss_history['train'],
                  'val_loss_history': loss_history['val'],
                  'timesteps': timesteps
              }, checkpoint_path)



def drawPDMloss(loss_history, epochs): # good
  model_name = 'probabilistic diffusion model'
  from sklearn.metrics import roc_curve, auc
  import matplotlib.pyplot as plot
  import seaborn as sns
  sns.set(style='whitegrid')
  loss_hist = loss_history
  sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["train"],label='Train loss',color='darkorange')
  sns.lineplot(x=[*range(1,epochs+1)],y=loss_hist["val"],label='Test loss',color='red')
  plt.title('Loss diagram of the '+ model_name)
  plt.ylabel('Loss value')
  plt.xlabel('Epoch')
  plt.show()

def loadModel(checkpoint_path,device):
  loss_history = {"train": [],"val": []}
  checkpoint = torch.load(checkpoint_path)
  model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,),
  )
  model.to(device)
  model.load_state_dict(checkpoint['model_state_dict'])
  # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  start_epoch = checkpoint['epoch']
  loss_history['train'] = checkpoint['train_loss_history']
  loss_history['val'] = checkpoint['val_loss_history']
  timesteps = checkpoint['timesteps']
  return {
      'loss_history':loss_history,
      'model':model,
      'epochs':start_epoch,
      'timesteps':timesteps
  }

def saveModelStates(model,savepath):
  torch.save(model.state_dict(),savepath)

def loadModelStates(savepath):
  return model.load_state_dict(torch.load(savepath))

def showTestResults(model,btfs_test,tensors_test,timesteps,generateSamples=3,gridW=4,gridH=4,figureSize=32,device='cuda'): #good
  """ generate images based on the test btfs and show the result """
  for id in range(len(btfs_test)):
    condit = btfs_test[id].to(device)
    btf_clone = condit.clone()
    for i in range(generateSamples-1):
        condit = torch.cat((condit,btf_clone), dim=0)
    samples = sample(model, image_size=image_size, timesteps=timesteps, batch_size=generateSamples, channels=channels,condition=condit)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(figureSize, figureSize))
    gs = gridspec.GridSpec(gridW, gridH)##
    for j in range(generateSamples):  ## 32
        ax = plt.Subplot(fig, gs[j])
        ax.imshow(samples[-1][j].reshape(image_size, image_size),cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
    plt.tight_layout()
    plt.show()
    plt.close()
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(2, 2)##
    ax = plt.Subplot(fig, gs[j])
    ax.imshow(tensors_test[id].cpu().reshape(image_size, image_size),cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)
    plt.tight_layout()
    plt.show()
    plt.close()

    import numpy
    import numpy as np
    from numpy import cov,trace,iscomplexobj,mean
    from scipy.linalg import sqrtm
    from keras.applications.inception_v3 import preprocess_input
    # FID function
    def calculate_fid(act1, act2):
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        ssdiff = numpy.sum((mu1 - mu2)**2.0)
        covmean = sqrtm(sigma1.dot(sigma2))
        if iscomplexobj(covmean): covmean = covmean.real
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid
    def getBest(lis):
      result = lis[0]
      m = abs(lis[0])
      for i in lis:
        if abs(i)<m:
          m = abs(i)
          result = i
      return result

    all_fid = []
    real = tensors_test[id].cpu().numpy().reshape(image_size, image_size)
    for i in range(generateSamples):
      generated = samples[-1][i].reshape(image_size, image_size)
      all_fid.append(calculate_fid(real, generated))

    print("best: ",getBest(all_fid))
    print('aveg: ',sum(all_fid)/generateSamples)


def showUnpairedResults(model,timesteps, generatedSamp=3,btfpath=None,testIds=None,gridnum=4,figureSize=32): # good
  """ generate images based on the unpaired btfs and show some result """
  if btfpath==None:
    btfpath = '/content/drive/MyDrive/End2End/Mu_all_BTFs.csv'
  df = pd.read_csv(btfpath)
  if testIds==None:
    ids = ['TCGA.A2.A04R','TCGA.A1.A0SG','TCGA.A2.A0CK','TCGA.A2.A0CO','TCGA.A2.A0CR']
  else:
    ids = testIds

  result = [] # normalized feature vectors
  for bt in ids:
      label = df.loc[df['ID']==bt]
      temp = label.drop(['ID'],axis=1).to_numpy().astype(np.float32)
      min_value = np.min(temp)
      max_value = np.max(temp)
      temp = (temp - min_value) / (max_value - min_value)
      temp = torch.from_numpy(temp).to(device)[:,None]
      result.append(temp)

  samp = generatedSamp
  for condit in result:
    btf_clone = condit.clone()
    for i in range(samp-1):
        condit = torch.cat((condit,btf_clone), dim=0)

    samples = sample(model, image_size=image_size, timesteps=timesteps, batch_size=samp, channels=channels,condition=condit)
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(figureSize, figureSize))
    gs = gridspec.GridSpec(gridnum, gridnum)
    for j in range(samp):
        ax = plt.Subplot(fig, gs[j])
        ax.imshow(samples[-1][j].reshape(image_size, image_size),cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
    plt.tight_layout()
    plt.show()
    plt.close()


def generateAllImages(model,timesteps,numGenerate=1,savePath=None,btfPath=None,start=0,end=1,gridnum=4,figureSize=32): # good
  """ Generate All Images """
  def generateI(condit, savePath=None):
    samples = sample(model, image_size=image_size, timesteps=timesteps, batch_size=numGenerate, channels=channels,condition=condit[1])
    if savePath==None:
      savePath = '/content/drive/MyDrive/End2End/generatedImg'
    path_volume = savePath+'/' + condit[0]
    if not os.path.exists(path_volume):
      os.makedirs(path_volume)
    for each in range(numGenerate):
      np.save(path_volume+'/{}.npy'.format(str(each).zfill(3)),samples[-1][each])

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(figureSize, figureSize))
    gs = gridspec.GridSpec(gridnum, gridnum)
    for j in range(numGenerate):
        ax = plt.Subplot(fig, gs[j])
        ax.imshow(samples[-1][j].reshape(image_size, image_size),cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
    plt.tight_layout()
    plt.show()
    plt.close()

  if btfPath==None:
    btfPath='/content/drive/MyDrive/End2End/Mu_all_BTFs.csv'
  df = pd.read_csv(btfPath)

  ids = df['ID'].tolist()
  result = []
  for bt in ids:
      label = df.loc[df['ID']==bt]
      temp = label.drop(['ID'],axis=1).to_numpy().astype(np.float32)
      min_value = np.min(temp)
      max_value = np.max(temp)
      temp = (temp - min_value) / (max_value - min_value)
      temp = torch.from_numpy(temp).to(device)[:,None]
      result.append([bt,temp])

  for each in range(len(result)):
    btf_clone = result[each][1].clone()
    for i in range(numGenerate-1):
        result[each][1] = torch.cat((result[each][1],btf_clone), dim=0)

  count = 0
  for condit in result:
      if count > end:
        break
      if count < start:
        count += 1
        continue
      print(count)
      count += 1
      generateI(condit,savePath)



""" Classification model """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import copy
import os
import torch
from PIL import Image
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torchvision import utils
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
torch.manual_seed(0)

import threading
class thread(threading.Thread):
  def __init__(self, func, args=()):
      super(thread, self).__init__()
      self.func = func
      self.args = args
  def run(self):
      self.result = self.func(*self.args)
  def get_result(self):
      try:
          return self.result
      except:
          return None

def npLoad(addr):
    return np.load(addr).astype(np.float32)

def separateData(ConfigC):
  from sklearn.model_selection import train_test_split # real images also split to test set, modify!
  test_size = ConfigC['test_size']

  # prepare Real MRIs
  realImgPath = ConfigC['realImgPath']
  def getMatchedImg(matchedImg) -> list: # use threading later
      result = []
      for dirpath, dirnames, filenames in os.walk(matchedImg):
          for filepath in filenames:
              result.append(os.path.join(dirpath, filepath))
      result.sort()  # modify here later
      return result
  realImgs = getMatchedImg(realImgPath)
  list_realImgs = []
  for pa in realImgs:
      img = Image.open(pa).convert('L')
      np_img = np.array(img)
      image = np.asarray([np_img], dtype=np.float32)
      list_realImgs.append(image)

  # prepare real image IDs
  realImgIDsPath = ConfigC['realImgIDsPath']
  realImgIDs = pd.read_csv(realImgIDsPath)
  realImgIDs = realImgIDs.values.tolist()
  list_realImgIDs = []
  for each in realImgIDs:
      list_realImgIDs.append(each[1].split('_')[0].replace('-','.'))

  # prepare mutation status labels
  mutationStatusPath = ConfigC['mutationStatusPath']
  mutationStatus = pd.read_csv(mutationStatusPath).drop(['Altered'],axis=1)
  realImgMutateStatus = pd.DataFrame(columns = ['ID', 'AKT1', 'ARID1A', 'BRCA1', 'CASP8', 'CBFB', 'CDH1', 'CDKN1B',
        'CHD4', 'CTCF', 'ERBB2', 'FBXW7', 'FOXA1', 'GATA3', 'GPS2', 'KMT2C',
        'KRAS', 'MAP2K4', 'MAP3K1', 'NCOR1', 'NF1', 'PIK3CA', 'PIK3R1', 'PTEN',
        'PTPRD', 'RB1', 'RUNX1', 'SF3B1', 'TBX3', 'TP53'])
  for each in list_realImgIDs:
    temp = mutationStatus.loc[mutationStatus['ID']==each]
    realImgMutateStatus = realImgMutateStatus.append(temp, ignore_index=True)

  # prepare generated images
  generatedImgPath = ConfigC['generatedImgPath']
  generatedImgIDs = os.listdir(generatedImgPath)
  btfPath = ConfigC['btfPath']
  numImgPerPatient = ConfigC['numImgPerPatient']
  btfs = pd.read_csv(btfPath)
  generatedImgNames = []
  print('Number of generated images:',len(generatedImgIDs))

  # for ids in btfs['ID'].tolist():   ###
  for ids in generatedImgIDs:
    filePath = generatedImgPath + '/' + ids + '/'
    for indx in range(numImgPerPatient):
      imgPath = filePath + str(indx).zfill(3) + '.npy'
      generatedImgNames.append(imgPath)
  list_generatedImgs = []
  threads = []
  for each in generatedImgNames:
      temp = thread(npLoad, (each,))
      threads.append(temp)
  for i in threads:
      i.start()
  for j in threads:
      j.join()
      list_generatedImgs.append(j.get_result())

  dict_generatedImgs = {}
  for each in range(len(list_generatedImgs)):
    dict_generatedImgs.update({generatedImgIDs[each]:list_generatedImgs[each]})
  dict_realImgs = {}
  for each in range(len(list_realImgIDs)):
    dict_realImgs.update({list_realImgIDs[each]:list_realImgs[each]})

  # prepare labels for generated images
  generatedImgMutateStatus = pd.DataFrame(columns = ['ID', 'AKT1', 'ARID1A', 'BRCA1', 'CASP8', 'CBFB', 'CDH1', 'CDKN1B',
        'CHD4', 'CTCF', 'ERBB2', 'FBXW7', 'FOXA1', 'GATA3', 'GPS2', 'KMT2C',
        'KRAS', 'MAP2K4', 'MAP3K1', 'NCOR1', 'NF1', 'PIK3CA', 'PIK3R1', 'PTEN',
        'PTPRD', 'RB1', 'RUNX1', 'SF3B1', 'TBX3', 'TP53'])
  # generatedImgIDs = btfs['ID'].tolist()
  for each in generatedImgIDs:
    temp = mutationStatus.loc[mutationStatus['ID']==each]
    generatedImgMutateStatus = generatedImgMutateStatus.append(temp, ignore_index=True)

  mode = ConfigC['mode']
  # clinical data
  if ConfigC['clinicalFlag']:
    clinicalDataPath = ConfigC['clinicalDataPath']
    clinical = pd.read_csv(clinicalDataPath)
    clinicalIDs = clinical['ID'].tolist()

    cGeneratedImgs = []
    cRealImgs = []
    generatedImgClinical = pd.DataFrame(columns = ['ID', 'er', 'pr', 'her2'])
    realImgClinical = pd.DataFrame(columns = ['ID', 'er', 'pr', 'her2'])
    for each in clinicalIDs:
      if each in generatedImgIDs:
        cGeneratedImgs.append(dict_generatedImgs[each])
        temp = clinical.loc[clinical['ID']==each]
        generatedImgClinical = generatedImgClinical.append(temp, ignore_index=True)
    for each in list_realImgIDs:
      if each in clinicalIDs:
        cRealImgs.append(dict_realImgs[each])
        temp = clinical.loc[clinical['ID']==each]
        realImgClinical = realImgClinical.append(temp, ignore_index=True)

    focusClinical = ConfigC['focusClinical']
    cGeneratedLabels = generatedImgClinical[focusClinical].tolist()
    cRealLabels = realImgClinical[focusClinical].tolist()
    print('clinical matched real:',len(cRealImgs))
    print('clinical matched generated:',len(cGeneratedImgs))
    if mode == 'mix':
      finalizedImgs = cRealImgs + cGeneratedImgs
      finalizedLabels = cRealLabels + cGeneratedLabels
    elif mode == 'real':
      finalizedImgs = cRealImgs
      finalizedLabels = cRealLabels
    elif mode == 'generated':
      finalizedImgs = cGeneratedImgs
      finalizedLabels = cGeneratedLabels
    else:
      print('Mode name error')
      raise
    print('total',len(finalizedImgs))
    mergedImgsAndLabels = pd.DataFrame({'images':finalizedImgs, 'labels':finalizedLabels})
    train, test = train_test_split(mergedImgsAndLabels, test_size=test_size, stratify=mergedImgsAndLabels['labels'])
    return train, test

  # finalize all Images and Labels
  focusGene = ConfigC['focusGene']
  finalizedImgs, finalizedLabels = [], []
  realImgMutateStatusLabels = realImgMutateStatus[focusGene].tolist()
  generatedImgMutateStatusLabels = generatedImgMutateStatus[focusGene].tolist()

  if mode == 'mix':
    finalizedImgs = list_realImgs + list_generatedImgs
    finalizedLabels = realImgMutateStatusLabels + generatedImgMutateStatusLabels
  elif mode == 'real':
    finalizedImgs = list_realImgs
    finalizedLabels = realImgMutateStatusLabels
  elif mode == 'generated':
    finalizedImgs = list_generatedImgs
    finalizedLabels = generatedImgMutateStatusLabels
  else:
    print('Mode name error')
    raise

  # separate data
  mergedImgsAndLabels = pd.DataFrame({'images':finalizedImgs, 'labels':finalizedLabels})
  train, test = train_test_split(mergedImgsAndLabels, test_size=test_size, stratify=mergedImgsAndLabels['labels'])

  return train, test

class loadDataC(Dataset):
    def __init__(self, dataset, dataType="train"):
      import torchvision.transforms as transforms
      from torchvision.transforms import ColorJitter
      self.train_transforms = transforms.Compose([
          transforms.ToPILImage(),
          transforms.RandomHorizontalFlip(),
          transforms.RandomVerticalFlip(),
          transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
          ColorJitter(contrast=[0.9, 1.1]),
          transforms.RandomRotation(10),
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,))
      ])
      self.test_transforms = transforms.Compose([
          transforms.ToPILImage(),
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,))
      ])
      self.dataType = dataType
      if self.dataType != "train" and self.dataType != "val":
        print('DataType Error!')
        raise
      self.finalizedImgs = dataset['images'].tolist()
      self.finalizedLabels = dataset['labels'].tolist()

    def __len__(self):
        if len(self.finalizedImgs) != len(self.finalizedLabels):
          print('Length Error!')
          raise
        return len(self.finalizedImgs)

    def __getitem__(self, idx):
      image = torch.from_numpy(self.finalizedImgs[idx].astype(np.float32))
      if self.dataType == "train":
        image = self.train_transforms(image)
      elif self.dataType == "val":
        image = self.test_transforms(image)
      return image, self.finalizedLabels[idx]

def get_dataloader(ConfigC):
  batch_size = ConfigC['batch_size']
  train, val = separateData(ConfigC)
  loadTrain = loadDataC(train,dataType="train")
  train_loader = DataLoader(loadTrain, batch_size=batch_size, shuffle=True)
  loadVal = loadDataC(val,dataType="val")
  val_loader = DataLoader(loadVal, batch_size=batch_size, shuffle=False)
  return loadTrain, loadVal, train_loader, val_loader


# torch.cuda.empty_cache()
""" origin train diffusion model """
SideImgPath = '../../SidePNG128'
SideBTFpath = '../../SideViewBTF.csv'
allBTFpath = '../../BTF_features.csv'
geneExprPath = '../../TCGABRCA_15gxp.csv'
mutationStatusPath = '../../all_genes.csv'
clinicalDataPath = '../../CliniUnOpHer2.csv'
savePath = '../../generatedImg_1'
btfPath = '../../allGeneExprs.csv'

image_size = 128
channels = 1
timesteps = 1500
loss_type = 'l2' # l1
lr = 1e-4
batch_size = 6  # 6
epochs = 1100  # 1100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = '../../geneExprModel_1.pt'

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,),
)
model.to(device)
from torch.optim import Adam
optimizer = Adam(model.parameters(), lr=lr)
use_checkpoint = False
torch.manual_seed(0)
btfs_train, btfs_test, tensors_train, tensors_test, status_train, status_test, clinicalStatus_train, clinicalStatus_test = prepareGeneDataset_end2end(SideImgPath,SideBTFpath,geneExprPath,mutationStatusPath,clinicalDataPath,gene,clinicalStatus)
train_loader, val_loader = get_dataloaderM_end2end(batch_size, btfs_train, btfs_test, tensors_train, tensors_test, status_train, status_test)
# train_loader, val_loader = get_dataloaderM_end2end(batch_size, btfs_train, btfs_test, tensors_train, tensors_test, clinicalStatus_train, clinicalStatus_test)
config = {
    'model':model,
    'optimizer':optimizer,
    'use_checkpoint':use_checkpoint,
    'checkpoint_path':checkpoint_path,
    'epochs':epochs,
    'train_loader':train_loader,
    'device':device,
    'loss_type':loss_type,
    'val_loader':val_loader,
    'timesteps':timesteps,
}

if sys.argv[1] == 'train':
    train_pdm(config)
    modelInf = loadModel(checkpoint_path, device)
    drawPDMloss(modelInf['loss_history'], modelInf['epochs'])
else:
    modelInf = loadModel(checkpoint_path,device)
    timesteps = modelInf['timesteps']
    model = modelInf['model']
    if sys.argv[1] == 'test':
        showTestResults(model,btfs_test,tensors_test,timesteps,generateSamples=3,gridW=4,gridH=4,figureSize=32,device='cuda')
    elif sys.argv[1] == 'unpaired':
        showUnpairedResults(model,timesteps,generatedSamp=3,btfpath=btfPath,testIds=None,gridnum=4,figureSize=32)
    elif sys.argv[1] == 'generate':
        generateAllImages(model,timesteps,numGenerate=1,savePath=savePath,btfPath=None,interrupt=4000,gridnum=2,figureSize=5)  #147
    else:
        print('Error mode')
        raise Exception
