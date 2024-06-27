'''
Author: error: git config user.name && git config user.email & please set dead value or install git
Date: 2023-01-17 09:53:38
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-02-06 00:26:03
FilePath: /Lingfeng He/xiongyali_new_idea/assignment/assign.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from utils.meters import AverageMeter
from optimizer import adjust_learning_rate
import time
import copy
from torch.autograd import Variable
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

def pairwise_distance(x, y):
   
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(x, y.t(), beta=1, alpha=-2)
        
    return dist_mat
    

class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, ratio=0.2):
        super(SoftCrossEntropyLoss, self).__init__()
        self.ratio = ratio

    def forward(self, pred, target):

        B = pred.shape[0]
        pred = torch.softmax(pred, dim=1)
        target = torch.softmax(target / self.ratio, dim=1).detach()

        loss = (-pred.log() * target).sum(1).sum() / B

        return loss
    

class OnlineRefinementLoss(nn.Module):
    def __init__(self, ratio=0.2):
        super(OnlineRefinementLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.sce_loss = SoftCrossEntropyLoss(ratio=ratio)
        self.temp = 0.05

    def forward(self, feat1, feat2, feat3, m):

        logit1 = F.normalize(feat1, dim=1).mm(m.features.t()) / self.temp
        logit2 = F.normalize(feat2, dim=1).mm(m.features.t()) / self.temp
        logit3 = F.normalize(feat3, dim=1).mm(m.features.t()) / self.temp  

        soft_logit1 = F.softmax(F.normalize(feat1, dim=1).mm(m.features.t()) / self.temp, dim=1).detach()
        soft_logit2 = F.softmax(F.normalize(feat2, dim=1).mm(m.features.t()) / self.temp, dim=1).detach()
        soft_logit3 = F.softmax(F.normalize(feat3, dim=1).mm(m.features.t()) / self.temp, dim=1).detach()

        loss = F.cross_entropy(logit1, (soft_logit2+soft_logit3)/2) +\
            F.cross_entropy(logit2, (soft_logit1+soft_logit3)/2) +\
            F.cross_entropy(logit3, (soft_logit1+soft_logit2)/2)
        
        loss /= 3
        return loss


def generate_one_hot_label(labels, num_cls):
    num_instance = labels.shape[0]
    one_hot_labels = torch.zeros(num_instance, num_cls).to(labels.device)
    one_hot_order = torch.arange(num_instance).to(labels.device)
    one_hot_labels[one_hot_order, labels] = 1
    
    return one_hot_labels

    

class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, predict, predict_):

        B = predict.shape[0]    
        predict = predict.clamp(min=1e-12) 
        predict_ = predict_.clamp(min=1e-12)     
        loss1 = ((predict * (predict.log() - predict_.detach().log())).sum(1).sum() / B)
        loss2 = ((predict_ * (predict_.log() - predict.detach().log())).sum(1).sum() / B)

        loss = (loss1 + loss2) / 2

        return loss
    

class BatchAlignmentLoss(nn.Module):
    def __init__(self):
        super(BatchAlignmentLoss, self).__init__()
        self.tau = 0.5

    def forward(self, feat_vp, feat_ap, feat_rp, label):

        feat_vp = F.normalize(feat_vp, dim=1)
        feat_ap = F.normalize(feat_ap, dim=1)
        feat_rp = F.normalize(feat_rp, dim=1)

        center_vp, center_ap, center_rp = [], [], []
        for y in label:
            center_vp.append(feat_vp[y==label].mean(0).unsqueeze(0))
            center_ap.append(feat_ap[y==label].mean(0).unsqueeze(0))
            center_rp.append(feat_rp[y==label].mean(0).unsqueeze(0))
        
        center_vp = F.normalize(torch.cat(center_vp, 0), dim=1)
        center_ap = F.normalize(torch.cat(center_ap, 0), dim=1)
        center_rp = F.normalize(torch.cat(center_rp, 0), dim=1)

        uni_label = label.unique()
        uni_center_vp, uni_center_ap, uni_center_rp = [], [], []
        for y in uni_label:
            uni_center_vp.append(feat_vp[y==label].mean(0).unsqueeze(0))
            uni_center_ap.append(feat_ap[y==label].mean(0).unsqueeze(0))
            uni_center_rp.append(feat_rp[y==label].mean(0).unsqueeze(0))
        
        uni_center_vp = F.normalize(torch.cat(uni_center_vp, 0), dim=1)
        uni_center_ap = F.normalize(torch.cat(uni_center_ap, 0), dim=1)
        uni_center_rp = F.normalize(torch.cat(uni_center_rp, 0), dim=1)

        loss_intra = ((feat_vp - center_vp) ** 2).sum(1).mean() +\
                    ((feat_ap - center_ap) ** 2).sum(1).mean() +\
                    ((feat_rp - center_rp) ** 2).sum(1).mean()
        
        center_label = torch.eye(uni_center_vp.shape[0]).to(uni_center_vp.device)
        loss_inter = -(torch.softmax(uni_center_vp.mm(uni_center_ap.t()) / self.tau, dim=1) * center_label).sum(1).log().mean() +\
                    -(torch.softmax(uni_center_vp.mm(uni_center_rp.t()) / self.tau, dim=1) * center_label).sum(1).log().mean() +\
                    -(torch.softmax(uni_center_ap.mm(uni_center_rp.t()) / self.tau, dim=1) * center_label).sum(1).log().mean()
        
        loss = (loss_intra + loss_inter)

        return loss
    


def union(a, b, c):
    out = 1 - (1-a) * (1-b) * (1-c)
    return out

        
class AssignTrainer(object):
    def __init__(self, args, encoder, batch_size, num_pos):
        
        super(AssignTrainer, self).__init__()
        self.encoder = encoder
        self.dim =2048
        self.part_num = args.part_num
        
        self.m_v = None
        self.m_r = None

        self.m_vp = None
        self.m_rp = None
        self.m_vpa = None
        self.m_rpa = None

        self.m_vpi = None
        self.m_rpi = None
        self.N_v = None

        self.m_vi = None
        self.m_ri = None

        self.label_v0, self.label_r0 = None, None
        
        self.temp = 0.05        
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.transform_to_image = transforms.Compose([
             transforms.ToPILImage()
        ])
        
        self.ce_loss = nn.CrossEntropyLoss()
        self.sce_loss = SoftCrossEntropyLoss()
        self.kl_loss = KLLoss()
        self.OLR_loss = OnlineRefinementLoss()
        self.batch_align_loss = BatchAlignmentLoss()

    def train(self, args, epoch, trainloader, optimizer, device, stage=None):
        current_lr = adjust_learning_rate(args, optimizer, epoch)

        batch_time = AverageMeter()
        contrast_rgb = AverageMeter()
        contrast_ir = AverageMeter()
        contrast_cross_rgb = AverageMeter()
        contrast_cross_ir = AverageMeter()
        OCLR_loss = AverageMeter()      
        Dp_loss = AverageMeter()  

        self.encoder.train()
        end = time.time()
                        
        print('epoch:{:5d}'.format(epoch))

        for batch_idx, (img10, img11, img2, label1, label2, cm_label1, cm_label2,\
                        y1, y2, cm_y1, cm_y2, y1_p, y2_p, cm_y1_p, cm_y2_p, idx1, idx2) in enumerate(trainloader):
            
            input10 = Variable(img10.to(device))
            input11 = Variable(img11.to(device))
            input2 = Variable(img2.to(device))    

            label1 = Variable(label1.to(device))
            label2 = Variable(label2.to(device))
            cm_label1 = Variable(cm_label1.to(device))
            cm_label2 = Variable(cm_label2.to(device))

            y1 = Variable(y1.to(device))
            y2 = Variable(y2.to(device))
            cm_y1 = Variable(cm_y1.to(device))
            cm_y2 = Variable(cm_y2.to(device))

            y1_p = Variable(y1_p.to(device))
            y2_p = Variable(y2_p.to(device))
            cm_y1_p = Variable(cm_y1_p.to(device))
            cm_y2_p = Variable(cm_y2_p.to(device))

            if stage == 'single':
                input1 = torch.cat((input10, input11), dim=0)
                feat1, feat2, feat1_pg, feat2_pg, _, _, _, _ = self.encoder(input1, input2)  
                single_size = feat1.shape[0] // 2
                feat10, feat11 = feat1[:single_size], feat1[single_size:]

                num_cls_rgb, num_cls_ir = self.m_v.features.shape[0], self.m_r.features.shape[0]
                pred10 = self.m_v(feat10, label1)
                pred11 = self.m_v(feat11, label1)
                pred2 = self.m_r(feat2, label2)
                loss_contrast_rgb = F.cross_entropy(pred10, label1) + F.cross_entropy(pred11, label1)
                loss_contrast_ir = F.cross_entropy(pred2, label2)
                loss_p = torch.tensor(0.).to(device)
                loss = loss_contrast_rgb + loss_contrast_ir

                contrast_rgb.update(loss_contrast_rgb.item())
                contrast_ir.update(loss_contrast_ir.item())
                Dp_loss.update(loss_p.item()) 

            elif stage == 'cross':
                
                idx1 = Variable(idx1.to(device))
                idx2 = Variable(idx2.to(device))

                num_cls_rgb, num_cls_ir = self.m_v.features.shape[0], self.m_r.features.shape[0]
                input1 = torch.cat((input10, input11), dim=0)
                feat1, feat2, feat1_pg, feat2_pg, feat1_pg1, feat2_pg1, feat1_pg2, feat2_pg2\
                = self.encoder(input1, input2)  

                single_size = feat1.shape[0] // 2
                batch_size = single_size * 3
                feat10, feat11 = feat1[:single_size], feat1[single_size:]
                feat10_pg, feat11_pg = feat1_pg[:single_size], feat1_pg[single_size:]
                feat10_pg1, feat11_pg1 = feat1_pg1[:single_size], feat1_pg2[single_size:]
                feat10_pg2, feat11_pg2 = feat1_pg2[:single_size], feat1_pg2[single_size:]

                feat = torch.cat((feat10, feat11, feat2), 0)
                feat_pg = torch.cat((feat10_pg, feat11_pg, feat2_pg), 0)
                feat_pg1 = torch.cat((feat10_pg1, feat11_pg1, feat2_pg1), dim=0)
                feat_pg2 = torch.cat((feat10_pg2, feat11_pg2, feat2_pg2), dim=0)

                label_v = torch.cat((label1, label1, cm_label2), 0)
                label_r = torch.cat((cm_label1, cm_label1, label2), 0)

                y_v = torch.cat((y1, y1, cm_y2), 0)
                y_r = torch.cat((cm_y1, cm_y1, y2), 0)

                y_p1 = torch.cat((y1, y1, y1, y1, y1, y1), 0)
                y_p2 = torch.cat((cm_y1, cm_y1, cm_y1, cm_y1, cm_y1, cm_y1), 0)
                y_p3 = torch.cat((cm_y2, cm_y2, cm_y2), 0)
                y_p4 = torch.cat((y2, y2, y2), 0)

                # ## logit for global features                
                # logit_v, logit_r = self.m_v(feat, label_v, cross=True), self.m_r(feat, label_r, cross=True)

                ## global contrastive loss
                if args.branch == 'dual':
                    logit_v, logit_r = self.m_v(feat, label_v, cross=True), self.m_r(feat, label_r, cross=True)
                    loss_g = F.cross_entropy(logit_v, y_v) + F.cross_entropy(logit_r, y_r) 
                elif args.branch == 'visible':
                    logit_v, logit_r = self.m_v(feat, label_v, cross=True), self.m_r(feat[2*single_size:], label_r[2*single_size:], cross=True)
                    loss_g = F.cross_entropy(logit_v, y_v) + F.cross_entropy(logit_r, y_r[2*single_size:]) 
                elif args.branch == 'infrared':
                    logit_v, logit_r = self.m_v(feat[:2*single_size], label_v[:2*single_size], cross=True), self.m_r(feat, label_r, cross=True)
                    loss_g = F.cross_entropy(logit_v, y_v[:2*single_size]) + F.cross_entropy(logit_r, y_r) 


                ## logit for part features
                logit_vp = torch.zeros(self.part_num, single_size*6, num_cls_rgb).to(device) 
                logit_rp = torch.zeros(self.part_num, single_size*6, num_cls_ir).to(device) 
                logit_vpa = torch.zeros(self.part_num, single_size*3, num_cls_rgb).to(device) 
                logit_rpa = torch.zeros(self.part_num, single_size*3, num_cls_ir).to(device)  

                for i in range(self.part_num):
                    feat_p = feat_pg[:, i*self.dim : (i+1)*self.dim]
                    feat_p1 = feat_pg1[:, i*self.dim : (i+1)*self.dim]
                    feat_p2 = feat_pg2[:, i*self.dim : (i+1)*self.dim]
                    feat_vp = torch.cat([feat_p[:2*single_size], feat_p1[:2*single_size], feat_p2[:2*single_size]], 0)
                    feat_rp = torch.cat([feat_p[2*single_size:], feat_p1[2*single_size:], feat_p2[2*single_size:]], 0)

                    logit_vp[i] += self.m_vp[i](feat_vp, torch.cat((label1,label1,label1,label1,label1,label1),0), cross=True)
                    logit_rp[i] += self.m_rp[i](feat_vp, torch.cat((cm_label1,cm_label1,cm_label1,cm_label1,cm_label1,cm_label1),0), cross=True)
                    logit_vpa[i] += self.m_vpa[i](feat_rp, torch.cat((cm_label2,cm_label2,cm_label2),0), cross=True)
                    logit_rpa[i] += self.m_rpa[i](feat_rp, torch.cat((label2,label2,label2),0), cross=True)

                ## part contrastive loss
                loss_pv = torch.tensor(0.).to(device)
                loss_pr = torch.tensor(0.).to(device)
                for i in range(self.part_num):
                    loss_pv += F.cross_entropy(logit_vp[i], y_p1) / self.part_num
                    loss_pv += F.cross_entropy(logit_rp[i], y_p2) / self.part_num
                    loss_pr += F.cross_entropy(logit_vpa[i], y_p3) / self.part_num
                    loss_pr += F.cross_entropy(logit_rpa[i], y_p4) / self.part_num

                ## instance contrastive loss
                k=args.k_instance
                logit_gi_v = torch.softmax(F.normalize(feat,dim=1).mm(self.m_vi.features.t()) / self.temp, dim=1)
                logit_gi_r = torch.softmax(F.normalize(feat,dim=1).mm(self.m_ri.features.t()) / self.temp, dim=1)

                rank_vg = torch.as_tensor((logit_gi_v) > (logit_gi_v).sort(dim=1, descending=True)[0][:,k].unsqueeze(1), dtype=torch.float)
                rank_rg = torch.as_tensor((logit_gi_r) > (logit_gi_r).sort(dim=1, descending=True)[0][:,k].unsqueeze(1), dtype=torch.float)
                rank_vg_refine, rank_rg_refine = torch.zeros_like(rank_vg).to(device), torch.zeros_like(rank_rg).to(device)

                ## cross-modality instance-level neighbor
                rank_vg_refine = rank_vg[:single_size] * rank_vg[single_size:2*single_size] * rank_vg[2*single_size:]
                rank_rg_refine = rank_rg[:single_size] * rank_rg[single_size:2*single_size] * rank_rg[2*single_size:]
                rank_vg_refine, rank_rg_refine = rank_vg_refine.repeat((3,1)), rank_rg_refine.repeat((3,1))

                ## instance part memory
                loss_pi = torch.tensor(0.).to(device)
                for i in range(self.part_num):
                    feat_p = feat_pg[:, i*self.dim : (i+1)*self.dim]
                    feat_p1 = feat_pg1[:, i*self.dim : (i+1)*self.dim]
                    feat_p2 = feat_pg2[:, i*self.dim : (i+1)*self.dim]
                    
                    # Vv, Va, Vr ...
                    feat_vp = torch.cat((feat_p[:single_size], feat_p2[single_size:2*single_size], feat_p1[2*single_size:]), dim=0)
                    feat_ap = torch.cat((feat_p2[:single_size], feat_p[single_size:2*single_size], feat_p2[2*single_size:]), dim=0)
                    feat_rp = torch.cat((feat_p1[:single_size], feat_p1[single_size:2*single_size], feat_p[2*single_size:]), dim=0)

                    feat_varp = F.normalize(torch.cat((feat_vp, feat_ap, feat_rp), dim=0), dim=1)

                    logit_pi_v = torch.softmax(feat_varp.mm(self.m_vpi[i].features.t()) / self.temp, dim=1)
                    logit_pi_r = torch.softmax(feat_varp.mm(self.m_rpi[i].features.t()) / self.temp, dim=1)
                    rank_vp = torch.as_tensor(logit_pi_v > logit_pi_v.sort(dim=1, descending=True)[0][:,k].unsqueeze(1), dtype=torch.float)
                    rank_rp = torch.as_tensor(logit_pi_r > logit_pi_r.sort(dim=1, descending=True)[0][:,k].unsqueeze(1), dtype=torch.float)

                    rank_vp1 = torch.cat((rank_vp[3*single_size:6*single_size], rank_vp[6*single_size:], rank_vp[:3*single_size]),0)
                    rank_vp2 = torch.cat((rank_vp[6*single_size:], rank_vp[:3*single_size], rank_vp[3*single_size:6*single_size]),0)
                    rank_rp1 = torch.cat((rank_rp[3*single_size:6*single_size], rank_rp[6*single_size:], rank_rp[:3*single_size]),0)
                    rank_rp2 = torch.cat((rank_rp[6*single_size:], rank_rp[:3*single_size], rank_rp[3*single_size:6*single_size]),0)
                    rank_vp = 1 - (1-rank_vp1) * (1-rank_vp2)
                    rank_rp = 1 - (1-rank_rp1) * (1-rank_rp2)
                    rank_vp = torch.cat((rank_vg, rank_vg, rank_vg), dim=0) * rank_vp
                    rank_rp = torch.cat((rank_rg, rank_rg, rank_rg), dim=0) * rank_rp

                    list_v, list_r = (rank_vp.sum(1) > 0), (rank_rp.sum(1) > 0)
                    loss_pi += (-(logit_pi_v * rank_vp)[list_v].sum(1).log()).mean()
                    loss_pi += (-(logit_pi_r * rank_rp)[list_r].sum(1).log()).mean()
                
                loss_pi /= self.part_num

                # instance global memory
                list_v, list_r = (rank_vg_refine.sum(1) > 0), (rank_rg_refine.sum(1) > 0)
                loss_gi = (-(logit_gi_v * rank_vg_refine)[list_v].sum(1).log()).mean()
                loss_gi += (-(logit_gi_r * rank_rg_refine)[list_r].sum(1).log()).mean()

                ## overall loss function
                if args.FGSAL == False and args.GPCR == False:
                    loss = loss_g

                elif args.FGSAL == True and args.GPCR == False:
                    loss = loss_g + (loss_pv + loss_pr)

                elif args.FGSAL == False and args.GPCR == True:
                    loss = loss_g + (loss_gi + loss_pi) * args.lambda_2

                elif args.FGSAL == True and args.GPCR == True:
                    loss = loss_g + (loss_pv + loss_pr) + (loss_gi + loss_pi) * args.lambda_2

                contrast_rgb.update(loss_g.item())
                contrast_ir.update(loss_g.item())   
                contrast_cross_rgb.update(loss_pv.item())
                contrast_cross_ir.update(loss_pr.item())
                OCLR_loss.update(loss_gi.item()) 
                Dp_loss.update(loss_pi.item()) 
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print log
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (batch_idx + 1) % args.print_step == 0:
                if stage == 'single':
                    print('Epoch: [{}][{}/{}] '
                        'lr:{:.8f} '
                        'contrast_RGB: {contrast_rgb.val:.4f}({contrast_rgb.avg:.3f}) '
                        'contrast_IR: {contrast_ir.val:.4f}({contrast_ir.avg:.3f}) '
                        'Dp_loss: {Dp_loss.val:.4f}({Dp_loss.avg:.3f}) '
                                                        .format(
                                                        epoch, batch_idx, len(trainloader), current_lr,
                                                        contrast_rgb = contrast_rgb,
                                                        contrast_ir = contrast_ir,
                                                        Dp_loss = Dp_loss))
                elif stage == 'cross':
                    print('Epoch: [{}][{}/{}] '
                        'lr:{:.8f} '
                        'MSLoss: {contrast_rgb.val:.4f}({contrast_rgb.avg:.3f}) '
                        'MALoss: {contrast_ir.val:.4f}({contrast_ir.avg:.3f}) '
                        '\nPartLoss: {contrast_cross_rgb.val:.4f}({contrast_cross_rgb.avg:.3f})'
                        'TotalLoss: {contrast_cross_ir.val:.4f}({contrast_cross_ir.avg:.3f})'
                        'OCLR_loss: {OCLR_loss.val:.4f}({OCLR_loss.avg:.3f})'
                        'Dp_loss: {Dp_loss.val:.4f}({Dp_loss.avg:.3f}) '.format(
                                                        epoch, batch_idx, len(trainloader), current_lr,
                                                        contrast_rgb = contrast_rgb,
                                                        contrast_ir = contrast_ir,
                                                        contrast_cross_rgb = contrast_cross_rgb,
                                                        contrast_cross_ir = contrast_cross_ir,
                                                        OCLR_loss = OCLR_loss,
                                                        Dp_loss = Dp_loss))   
                
                else:
                    print('-----Wrong stage-----')


            
            
            