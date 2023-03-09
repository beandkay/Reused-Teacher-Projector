# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from DisCo.models.layers.activations_me import HardMishMe
from DisCo.models.efficientnet_blocks import InvertedResidual

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
def conv3x3(in_channels, out_channels, stride=1, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
        
class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, teacher_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, args=None, factor=2):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders

	# for DLA/EfficientNet
        self.encoder_q = base_encoder(num_classes=dim, pretrained=False)
        self.encoder_k = base_encoder(num_classes=dim, pretrained=False)

        # num_classes is the output fc dimension
        if args.teacher_arch in ['resnet50', 'resnet101', 'resnet152']:
            self.teacher_encoder_q = teacher_encoder(num_classes=dim)
            # self.teacher_encoder_k = teacher_encoder(num_classes=dim)
        elif args.teacher_arch == 'resnet50w2':
            self.teacher_encoder_q = teacher_encoder(
                normalize=True,
                hidden_mlp=8192,
                output_dim=dim,
                nmb_prototypes=args.nmb_prototypes
            )
            # self.teacher_encoder_k = teacher_encoder(
            #     normalize=True,
            #     hidden_mlp=8192,
            #     output_dim=dim,
            #     nmb_prototypes=args.nmb_prototypes
            # )
        elif args.teacher_arch in ['SWAVresnet50', 'DCresnet50', 'SELAresnet50']:
            self.teacher_encoder_q = teacher_encoder(
                normalize=True,
                hidden_mlp=2048,
                output_dim=dim,
                nmb_prototypes=args.nmb_prototypes
            )
            # self.teacher_encoder_k = teacher_encoder(
            #     normalize=True,
            #     hidden_mlp=2048,
            #     output_dim=dim,
            #     nmb_prototypes=args.nmb_prototypes
            # )

        # teacher mlp
        if mlp:
            if args.teacher_arch in ['resnet50', 'resnet101', 'resnet152']:
                dim_mlp = self.teacher_encoder_q.fc.weight.shape[1]
                self.teacher_encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.teacher_encoder_q.fc)
                # self.teacher_encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.teacher_encoder_k.fc)
        self.t_n = dim_mlp
        
        # student mlp
        if mlp:
            if args.arch.startswith('efficient') or args.arch == 'mobilenetv3':
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                # self.encoder_q.classifier = nn.Sequential(nn.Linear(dim_mlp, 2048), nn.ReLU(), nn.Linear(2048, 128))
                # self.encoder_k.classifier = nn.Sequential(nn.Linear(dim_mlp, 2048), nn.ReLU(), nn.Linear(2048, 128))
                self.encoder_q.classifier = self.encoder_k.classifier = self.teacher_encoder_q.fc
            else:
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                #tmpq = self.encoder_q.classifier
                #tmpk = self.encoder_k.classifier
                # self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, 2048), nn.ReLU(), nn.Linear(2048, 128))
                # self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, 2048), nn.ReLU(), nn.Linear(2048, 128))
                self.encoder_q.fc = self.encoder_k.fc = self.teacher_encoder_q.fc
        self.s_n = dim_mlp

        # teacher gard
        # for param_q, param_k in zip(self.teacher_encoder_q.parameters(), self.teacher_encoder_k.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient
        #     param_q.requires_grad = False
        
        for param_q in self.teacher_encoder_q.parameters():
            param_q.requires_grad = False

        # student grad
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data) # initialize
            param_k.requires_grad = False  # not update by gradient
            param_q.requires_grad = True
            
        for (name_param_q, param_q), (name_param_k, param_k) in zip(self.encoder_q.named_parameters(), self.encoder_k.named_parameters()):
            if 'fc' in name_param_q or 'classifier' in name_param_q:
                param_q.requires_grad = False
            if 'fc' in name_param_k or 'classifier' in name_param_k:
                param_k.requires_grad = False
        
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))   

        # A bottleneck design to reduce extra parameters
        self.encoder_q.transfer = self.encoder_k.transfer = nn.Sequential(
            conv1x1(self.s_n, self.t_n//factor),
            nn.BatchNorm2d(self.t_n//factor),
            nn.ReLU(inplace=True),
            conv3x3(self.t_n//factor, self.t_n//factor),
            # depthwise convolution
            #conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
            nn.BatchNorm2d(self.t_n//factor),
            nn.ReLU(inplace=True),
            conv1x1(self.t_n//factor, self.t_n),
            nn.BatchNorm2d(self.t_n),
            nn.ReLU(inplace=True),
            ) if args.arch in ['resnet50', 'resnet101', 'resnet152'] \
        else InvertedResidual(self.s_n, self.t_n, act_layer=HardMishMe, norm_layer=nn.BatchNorm2d, exp_ratio=(self.t_n//self.s_n))
        
    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        #print ("query images shape:", im_q.shape)
        #print ("key images shape:", im_k.shape)

        # compute query features
        feat_q = self.encoder_q.forward_features(im_q)
        # q = self.encoder_q.fc(feat_q)  # queries: NxC
        # q = F.normalize(q, dim=1)

        feat_qk = self.encoder_q.forward_features(im_q)
        # qk = self.encoder_q.fc(feat_qk)
        # qk = F.normalize(qk, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            teacher_qk = self.teacher_encoder_q(im_k)
            teacher_qk = F.normalize(teacher_qk, dim=1)

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            
            feat_k = self.encoder_k.forward_features(im_k)  # keys: NxC

            #teacher_k = self.teacher_encoder_k(im_k)
            #teacher_k = F.normalize(teacher_k, dim=1)
            #teacher_k = self._batch_unshuffle_ddp(teacher_k, idx_unshuffle)

            teacher_q = self.teacher_encoder_q(im_q)  # queries: NxC
            teacher_q = F.normalize(teacher_q, dim=1)
            
        # Prediction via Teacher Classifier
        temp_feat_q = self.teacher_encoder_q.avgpool(feat_q)
        temp_feat_q = temp_feat_q.view(temp_feat_q.size(0), -1)
        q = self.encoder_q.get_classifier()(temp_feat_q)
        q = F.normalize(q, dim=1)
        
        temp_feat_qk = self.teacher_encoder_q.avgpool(feat_qk)
        temp_feat_qk = temp_feat_q.view(temp_feat_qk.size(0), -1)
        qk = self.encoder_k.get_classifier()(temp_feat_qk)
        qk = F.normalize(qk, dim=1)
        
        with torch.no_grad():
            temp_feat_k = self.teacher_encoder_q.avgpool(feat_k)
            temp_feat_k = temp_feat_k.view(temp_feat_k.size(0), -1)
            k = self.encoder_k.get_classifier()(temp_feat_k)
            k = F.normalize(k, dim=1)
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, q, teacher_q, qk, teacher_qk


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
