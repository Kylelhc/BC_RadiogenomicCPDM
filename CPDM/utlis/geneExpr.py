import torch, os, sys, warnings, math, random
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
# import matplotlib.pyplot as plt
from inspect import isfunction
from functools import partial
from tqdm.auto import tqdm
from torch import einsum
from torch.optim import Adam

warnings.filterwarnings("ignore")
seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

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

def prepareGeneDataset(SideImgPath,SideImgIDpath,geneExprPath,device, test = [1,16,24,33]):
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

  # mutationStatus = pd.read_csv(mutationStatusPath).drop(['Altered'],axis=1)
  # realImgMutateStatus = pd.DataFrame(columns = ['ID', 'AKT1', 'ARID1A', 'BRCA1', 'CASP8', 'CBFB', 'CDH1', 'CDKN1B',
  #       'CHD4', 'CTCF', 'ERBB2', 'FBXW7', 'FOXA1', 'GATA3', 'GPS2', 'KMT2C',
  #       'KRAS', 'MAP2K4', 'MAP3K1', 'NCOR1', 'NF1', 'PIK3CA', 'PIK3R1', 'PTEN',
  #       'PTPRD', 'RB1', 'RUNX1', 'SF3B1', 'TBX3', 'TP53'])
  # for each in sideViewIDs:
  #   temp = mutationStatus.loc[mutationStatus['ID']==each]
  #   realImgMutateStatus = realImgMutateStatus.append(temp, ignore_index=True)

  # status = realImgMutateStatus[gene].tolist()

  #### clinical
  # clinical = pd.read_csv(clinicalDataPath)
  # realImgClinical = pd.DataFrame(columns = ['ID', 'er', 'pr', 'her2'])
  # for each in sideViewIDs:
  #   temp = clinical.loc[clinical['ID']==each]
  #   realImgClinical = realImgClinical.append(temp, ignore_index=True)
  # clinicalStatus = realImgClinical[cParam].tolist()
  ####

  btfs_train = []
  btfs_test = []
  tensors_train = []
  tensors_test = []
  # status_train = []
  # status_test = []
  # clinicalStatus_train = []
  # clinicalStatus_test = []

  # test = [1,5,7,12,16,19,24,33]
  # test = [1,16,24,33]
  for te in range(len(tensors)):
    if te in test:
      btfs_test.append(btfs[te])
      tensors_test.append(tensors[te])
      # status_test.append(status[te])
      # clinicalStatus_test.append(clinicalStatus[te])
    else:
      btfs_train.append(btfs[te])
      tensors_train.append(tensors[te])
      # status_train.append(status[te])
      # clinicalStatus_train.append(clinicalStatus[te])

  return btfs_train, btfs_test, tensors_train, tensors_test#, status_train, status_test, clinicalStatus_train, clinicalStatus_test


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


def get_dataloader(batch_size, btfs_train, btfs_test, tensors_train, tensors_test):
  load_train_set = load_data(images=tensors_train, btfs=btfs_train)
  train_loader = DataLoader(load_train_set, batch_size=batch_size, shuffle=True)
  load_val_set = load_data(images=tensors_test, btfs=btfs_test, data_type="val")
  val_loader = DataLoader(load_val_set, batch_size=batch_size, shuffle=False)
  # return load_train_set, load_val_set, train_loader, val_loader
  return train_loader, val_loader


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

  use_checkpoint = config['use_checkpoint']
  checkpoint_path = config['checkpoint_path']
  epochs = config['epochs']
  loss_type = config['loss_type']
  timesteps = config['timesteps']
  image_size = config['image_size']
  channels = config['channels']
  batch_size = config['batch_size']
  lr = config['lr']
  SideImgPath = config['SideImgPath']
  SideImgIDpath = config['SideImgIDpath']
  geneExprPath = config['geneExprPath']
  test = config['test']
  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,),
  )
  if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
  model = model.to(device)
  optimizer = Adam(model.parameters(), lr=lr)

  if use_checkpoint:
    mname = checkpoint_path.strip('.pt')
    num = str(int(mname[-1])+1)
    mname = mname[:-1]
    checkpoint_path=mname+num+'.pt'
  print('Saved:', checkpoint_path)

  btfs_train, btfs_test, tensors_train, tensors_test = prepareGeneDataset(SideImgPath, SideImgIDpath, geneExprPath, device, test)
  train_loader, val_loader = get_dataloader(batch_size, btfs_train, btfs_test, tensors_train, tensors_test)

  loss_history = {"train": [],"val": []}
  if os.path.isfile(checkpoint_path) and use_checkpoint:
      checkpoint = torch.load(checkpoint_path)
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      start_epoch = checkpoint['epoch']
      loss_history['train'] = checkpoint['train_loss_history']
      loss_history['val'] = checkpoint['val_loss_history']
      timesteps = checkpoint['timesteps']
      loss_type = checkpoint['loss_type']
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
                  'timesteps': timesteps,
                  'loss_type': loss_type
              }, checkpoint_path)

























