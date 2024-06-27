import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .backbone.resnet import resnet50
from .backbone.pooling import build_pooling_layer
import copy


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_out")
        init.zeros_(m.bias.data)
    elif classname.find("BatchNorm1d") != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias is not None:
            init.zeros_(m.bias.data)
            
            
class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=8):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)



        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

            
            
class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x



class gradientreverselayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, coeff, input):
        ctx.coeff = coeff
        # this is necessary. if we just return "input", "backward" will not be called sometimes
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        coeff = ctx.coeff
        return None, -coeff * grad_outputs


class AdversarialLayer(nn.Module):
    def __init__(self, per_add_iters, iter_num=0, alpha=10.0, low_value=0.0, high_value=1.0, max_iter=10000.0):
        super(AdversarialLayer, self).__init__()
        self.per_add_iters = per_add_iters
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter
        self.grl = gradientreverselayer.apply

    def forward(self, input, train_set=True):
        if train_set:
            self.iter_num += self.per_add_iters
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                        self.high_value - self.low_value) + self.low_value)

        return self.grl(self.coeff, input)


class DiscriminateNet(nn.Module):
    def __init__(self, input_dim, class_num=1):
        super(DiscriminateNet, self).__init__()
        self.ad_layer1 = nn.Linear(input_dim, input_dim//2)
        self.ad_layer2 = nn.Linear(input_dim//2, input_dim//2)
        self.ad_layer3 = nn.Linear(input_dim//2, class_num)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.bn = nn.BatchNorm1d(class_num)
        self.bn2 = nn.BatchNorm1d(input_dim // 2)
        self.bn.bias.requires_grad_(False)
        self.bn2.bias.requires_grad_(False)
        self.sigmoid = nn.Sigmoid()

        self.ad_layer1.apply(weights_init_kaiming)
        self.ad_layer2.apply(weights_init_kaiming)
        self.ad_layer3.apply(weights_init_classifier)

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.bn(x)
        x = self.sigmoid(x)  # binary classification

        return x



class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class SAFL(nn.Module):
    def __init__(self, dim = 2048, part_num = 6) -> None:
        super().__init__()

        self.part_num = part_num
        self.part_tokens = nn.Parameter(nn.init.kaiming_normal_(torch.empty(part_num, 2048)))
        self.pos_embeding = nn.Parameter(nn.init.kaiming_normal_(torch.empty(18 * 9, dim)))
        self.pos_embeding.data.copy_(torch.zeros(18 * 9, dim))

        self.active = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0,2,1) # [B, HW, C]
        num_patch = x.shape[1]
        out = []
        for i in range(self.part_num):
            x_p = x[:, (num_patch//self.part_num)*i:(num_patch//self.part_num)*(i+1), :]
            attn = self.part_tokens[i].unsqueeze(0) @ x_p.transpose(-1, -2)
            attn = self.active(attn)
            attn = attn / attn.sum(-1).unsqueeze(-1)
            x_p = attn @ x_p
            out.append(x_p)
        out = torch.cat(out, 1)

        return out, attn

    

class QueryAttention(nn.Module):
    def __init__(self, dim = 2048, part_num = 6) -> None:
        super().__init__()
        self.dim = dim
        self.part_num = part_num
    
    def forward(self, x, part_feat): ## query: b, p, c
        B, C, H, W = x.shape
        num_patch = H*W
        x = x.reshape(B, C, -1)
        out = []
        attn_out = []
        for i in range(self.part_num):
            x_p = x[:, :, (num_patch//self.part_num)*i : (num_patch//self.part_num)*(i+1)] ## b, c, h*w/p
            attn = torch.softmax(part_feat[:, i].unsqueeze(1).bmm(x_p) / (self.dim ** 0.5), dim=-1) ## b, 1, h*w/p
            x_p = attn.bmm(x_p.permute(0,2,1)) ## b, 1, c
            out.append(x_p)
            attn_out.append(attn)
        out = torch.cat(out, 1)
        attn_out = torch.cat(attn_out, 1)

        return out, attn_out
    


class BaseResNet(nn.Module):
    def __init__(self, args, class_num, non_local= 'on', gm_pool = 'on', per_add_iters=1, arch='resnet50'):
        super(BaseResNet, self).__init__()
        
        pool_dim = 2048
        self.part_num = args.part_num

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)

        self.attn_pool = SAFL(dim=pool_dim, part_num=self.part_num)
        self.query_attn = QueryAttention(dim=pool_dim, part_num=self.part_num)
        
        self.adnet = AdversarialLayer(per_add_iters=per_add_iters)
        self.disnet = DiscriminateNet(pool_dim, 2)

        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_part = nn.BatchNorm1d(2048 * self.part_num)
        self.bottleneck_part.bias.requires_grad_(False)  # no shift
        self.bottleneck_part.apply(weights_init_kaiming)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.non_local = non_local
        if self.non_local =='on':
            layers=[3, 4, 6, 3]
            non_layers=[0,2,3,0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        self.class_num = class_num
        if self.class_num > 0:
            self.classifier = nn.Linear(pool_dim, class_num, bias=False)
            self.classifier.apply(weights_init_classifier)
            
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool
        self.gem = build_pooling_layer('gem')
        self.sigmoid = nn.Sigmoid()
        self.mode='RGB'
        self.classifier = nn.Linear(2048, 395, bias=False)
        self.classifier_v = nn.Linear(2048, 805, bias=False)
        self.classifier_r = nn.Linear(2048, 424, bias=False)

        self.softmax = nn.Softmax()
        self.part = 0

        self.cross_modality_tensor=None
        
    def forward(self, x_rgb, x_ir=None, mode=None):

        mode = self.mode

        if mode == None:
            x_rgb_ = self.visible_module(self.cross_modality_tensor)
            x_ir_ = self.thermal_module(x_rgb)
            x = torch.cat((x_rgb_, x_ir_), dim=0)
            
            feat_g, feat_pg, feat_pg1 = self.forward_main_net(x, mode=0)
            
            n_rgb = x_rgb.shape[0]
            feat_rgb_pg1, feat_ir_pg1 = feat_pg1[:n_rgb], feat_pg1[n_rgb:]
    
            return self.classifier_v(F.normalize(feat_ir_pg1[:,(self.part-1)*2048:self.part*2048], dim=1))
            # return self.classifier_v(F.normalize(feat_rgb_pg1[:,(self.part-1)*2048:self.part*2048], dim=1))

        
        if mode == 'RGB':
            x_rgb = self.visible_module(x_rgb)
            feat_rgb_g, feat_rgb_pg = self.forward_main_net(x_rgb, mode=1)
            if self.part == 0:
                return self.classifier(feat_rgb_g)
            else:
                return self.classifier(feat_rgb_pg[:,(self.part-1)*2048:self.part*2048])
        
        if mode == 'IR':
            x_ir = self.thermal_module(x_rgb)
            feat_ir_g, feat_ir_pg = self.forward_main_net(x_ir, mode=2)
            if self.part == 0:
                return self.classifier(feat_ir_g)
            else:
                return self.classifier(feat_ir_pg[:,(self.part-1)*2048:self.part*2048])

    def forward_main_net(self, x, mode=0):
                
        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet.base.layer1(x)
            x = self.base_resnet.base.layer2(x)
            x = self.base_resnet.base.layer3(x)
            x = self.base_resnet.base.layer4(x)
        
        global_feat = x
        b = global_feat.shape[0]
        part_feat, _ = self.attn_pool(global_feat)

        single_size = part_feat.shape[0] // 2
        global_feat_v, global_feat_r = global_feat[:single_size], global_feat[single_size:]
        part_feat1, _ = self.query_attn(torch.cat((global_feat_r, global_feat_v), dim=0), part_feat)
        part_feat, _ = self.query_attn(global_feat, part_feat)
        
        feat_p = self.bottleneck_part(part_feat.reshape(x.shape[0], -1))
        feat_p1 = self.bottleneck_part(part_feat1.reshape(x.shape[0], -1))
            
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_g = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)

        else:
            x_g = self.avgpool(x)
            x_g = x_g.view(x_g.size(0), x_g.size(1))

        feat_g = self.bottleneck(x_g)
        feat_pg = torch.cat((feat_p, feat_g), dim=1)
        # if self.training:
        feat_pg1 = torch.cat((feat_p1, feat_g), dim=1)
        
        return feat_g, feat_pg, feat_pg1
        # if self.training:
        #     return feat_g, feat_pg, feat_pg1
        # else:
        #     return F.normalize(feat_g, dim=1), F.normalize(feat_pg, dim=1)