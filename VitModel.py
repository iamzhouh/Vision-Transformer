# 参考 https://blog.csdn.net/weixin_44966641/article/details/118733341

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torchsummary import summary

# 作用是：判断t是否是元组，如果是，直接返回t；如果不是，则将t复制为元组(t, t)再返回。
# 用来处理当给出的图像尺寸或块尺寸是int类型（如224）时，直接返回为同值元组（如(224, 224)）。
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# PreNorn对应框图中最下面的黄色的Norm层。其参数dim是维度，而fn则是预先要进行的处理函数，是Attention、FeedForward之一
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))


# FeedForward层由全连接层，配合激活函数GELU和Dropout实现，对应框图中蓝色的MLP。
# 参数dim和hidden_dim分别是输入输出的维度和中间层的维度，dropour则是dropout操作的概率参数p。
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


# Attention，Transformer中的核心部件，对应框图中的绿色的Multi-Head Attention。
# 参数heads是多头自注意力的头的数目，dim_head是每个头的维度。
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head = 64, dropout = 0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)    # (b, n(65), dim*3) ---> 3 * (b, n, dim)  按列切开
        q, k ,v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        # q, k, v   (b, h, n, dim_head(64))  把q,k,v分别分为h个（h为head个数）
        # q, k ,v = rearrange(qkv, 'b n (h d) -> b h n d', h=h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale  # 矩阵乘法

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)   # 矩阵乘法
        out = rearrange(out, 'b h n d -> b n (h d)')  # 将 head 合并
        return self.to_out(out)

# 可以构建整个Transformer Block了，即对应框图中的整个右半部分Transformer Encoder。有了前面的铺垫，整个Block的实现看起来非常简洁。
# 参数depth是每个Transformer Block重复的次数，其他参数与上面各个层的介绍相同。
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert  image_height % patch_height ==0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					# nn.Parameter()定义可学习参数
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)        # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape           # b表示batchSize, n表示每个块的空间分辨率, _表示一个块内有多少个值

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)               # 将cls_token拼接到patch token中去       (b, 65, dim)
        x += self.pos_embedding[:, :(n+1)]                  # 加位置嵌入（直接加）      (b, 65, dim)
        x = self.dropout(x)

        x = self.transformer(x)                                                 # (b, 65, dim)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)

        x = self.to_latent(x)                                                   # Identity (b, dim)
        print(x.shape)

        return self.mlp_head(x)                                                 #  (b, num_classes)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_vit = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    model_vit.to(device)
    summary(model_vit, input_size=(3, 256, 256))