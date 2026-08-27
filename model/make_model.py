import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # ================== 【新增：安全获取配置】 ==================
        cls_sep_flag = getattr(cfg.MODEL, 'CLS_SEP', False)
        cls_gen_type = getattr(cfg.MODEL, 'CLS_GEN_TYPE', 'dynamic')
        cls_mlp_ratio = getattr(cfg.MODEL, 'CLS_MLP_RATIO', 4.0)  
        use_rope_flag = getattr(cfg.MODEL, 'USE_ROPE', False)
        abs_pos_mode_flag = getattr(cfg.MODEL, 'ABS_POS_MODE', 'normal')
        # ==========================================================
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE,
                                                        cls_sep=cls_sep_flag,
                                                        cls_gen_type=cls_gen_type,
                                                        cls_mlp_ratio=cls_mlp_ratio,
                                                        use_rope=use_rope_flag,
                                                        abs_pos_mode=abs_pos_mode_flag
                                                        )
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # ================== 【新增：安全获取配置】 ==================
        cls_sep_flag = getattr(cfg.MODEL, 'CLS_SEP', False)
        cls_gen_type = getattr(cfg.MODEL, 'CLS_GEN_TYPE', 'dynamic')
        cls_mlp_ratio = getattr(cfg.MODEL, 'CLS_MLP_RATIO', 4.0)
        use_rope_flag = getattr(cfg.MODEL, 'USE_ROPE', False)
        abs_pos_mode_flag = getattr(cfg.MODEL, 'ABS_POS_MODE', 'normal')
        # ==========================================================
        
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, 
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        cls_sep=cls_sep_flag,
                                                        cls_gen_type=cls_gen_type,
                                                        cls_mlp_ratio=cls_mlp_ratio,
                                                        use_rope=use_rope_flag,
                                                        abs_pos_mode=abs_pos_mode_flag
                                                        )
        
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

    def forward(self, x, label=None, cam_label= None, view_label=None):  # label is unused if self.cos_layer == 'no'
        # ===============================================
        r_cos = getattr(self.base, 'rope_cos', None)
        r_sin = getattr(self.base, 'rope_sin', None)

        if getattr(self, 'cls_sep', False):
            # 解包主干传来的 3 个组件
            features, cls_outputs_11, query_12_norm = self.base(x, cam_label=cam_label, view_label=view_label)
            
            # 1. 全局分支
            b1_feat = self.b1(features, r_cos, r_sin)  # 执行第 12 层, 输出 [B, 196, 768]
            b1_feat_norm = self.base.cross_norm_x_list[-1](b1_feat)
            
            K_global = b1_feat_norm
            if getattr(self.base, 'abs_pos_mode', 'normal') == 'cross_attn':
                K_global = K_global + self.base.pos_embed[:, 1:, :]
                
            # 使用 query_12 收集全局特征
            attn_g = (query_12_norm @ K_global.transpose(-2, -1)) * self.scale   # [B, 1, C] @ [B, C, N] -> [B, 1, N] = [B, 1, 196]
            cls_12_global = attn_g.softmax(dim=-1) @ b1_feat_norm    # [B, 1, N] @ [B, N, C] -> [B, 1, C] = [B, 1, 768]
            
            # 拼接 (11 + 1 = 12)，交由全局聚合器 (复用 base 里的 cls_aggregator)
            global_stacked = torch.cat([cls_outputs_11, cls_12_global], dim=1).transpose(1, 2)  # [B, 768, 12]
            global_feat = self.base.cls_aggregator(global_stacked).squeeze(-1)  # [B, 768]

            # 2. JPM 局部分支
            if r_cos is not None:
                # 增加 batch 维度以适应 shuffle_unit 
                r_cos_shuf = r_cos.unsqueeze(0) # 变成 [1, N, C]
                r_sin_shuf = r_sin.unsqueeze(0)
            else:
                r_cos_shuf, r_sin_shuf = None, None
            
            if getattr(self.base, 'abs_pos_mode', 'normal') == 'cross_attn':
                pos_shuf = self.base.pos_embed[:, 1:, :] 
            else:
                pos_shuf = None  

            if self.rearrange:
                x_local = shuffle_unit(features, self.shift_num, self.shuffle_groups, begin=0)
                if r_cos is not None:
                    # 使用与 features 完全一致的参数打乱坐标！
                    r_cos_shuf = shuffle_unit(r_cos_shuf, self.shift_num, self.shuffle_groups, begin=0)
                    r_sin_shuf = shuffle_unit(r_sin_shuf, self.shift_num, self.shuffle_groups, begin=0)
                if pos_shuf is not None:
                    pos_shuf = shuffle_unit(pos_shuf, self.shift_num, self.shuffle_groups, begin=0)
            else:
                x_local = features
            
            patch_length = x_local.size(1) // self.divide_length
            local_feats = []
            
            for i in range(4):
                if r_cos is not None:
                    # 提取对应区间的打乱后的坐标
                    rc_i = r_cos_shuf[0, i*patch_length : (i+1)*patch_length]
                    rs_i = r_sin_shuf[0, i*patch_length : (i+1)*patch_length]
                else:
                    rc_i, rs_i = None, None

                # 切片并执行第 12 层
                b2_feat = self.b2(x_local[:, i*patch_length : (i+1)*patch_length], rc_i, rs_i)
                b2_feat_norm = self.base.local_cross_norm_x(b2_feat)
                
                K_local = b2_feat_norm
                if pos_shuf is not None:
                    K_local = K_local + pos_shuf[:, i*patch_length : (i+1)*patch_length]

                # 使用相同的 query_12 收集局部组特征
                attn_l = (query_12_norm @ K_local.transpose(-2, -1)) * self.scale
                cls_12_local = attn_l.softmax(dim=-1) @ b2_feat_norm  # [B, 1, 768]
                
                # 拼接 (11 + 1 = 12)，交由【局部共享聚合器】
                local_stacked = torch.cat([cls_outputs_11, cls_12_local], dim=1).transpose(1, 2) # [B, 768, 12]
                local_feat = self.base.local_cls_aggregator(local_stacked).squeeze(-1)  # [B, 768]
                
                local_feats.append(local_feat)
                
            local_feat_1, local_feat_2, local_feat_3, local_feat_4 = local_feats

        else:
            # 原版 JPM 逻辑 (保持不变)
            features = self.base(x, cam_label=cam_label, view_label=view_label)

            # =========为带有 cls_token 的分支补齐 0 旋转 ================
            if r_cos is not None:
                r_cos_f = torch.cat([torch.ones(1, r_cos.shape[-1], device=x.device), r_cos], dim=0)
                r_sin_f = torch.cat([torch.zeros(1, r_sin.shape[-1], device=x.device), r_sin], dim=0)
            else:
                r_cos_f, r_sin_f = None, None
            # ===========================================================

            # global branch
            b1_feat = self.b1(features, r_cos_f, r_sin_f) 
            global_feat = b1_feat[:, 0]

            feature_length = features.size(1) - 1
            patch_length = feature_length // self.divide_length
            token = features[:, 0:1]

            if r_cos is not None:
                r_cos_shuf = r_cos.unsqueeze(0)
                r_sin_shuf = r_sin.unsqueeze(0)
            else:
                r_cos_shuf, r_sin_shuf = None, None

            if self.rearrange:
                x = shuffle_unit(features, self.shift_num, self.shuffle_groups)  # 默认 begin=1
                if r_cos is not None:
                    r_cos_shuf = shuffle_unit(r_cos_shuf, self.shift_num, self.shuffle_groups, begin=0)
                    r_sin_shuf = shuffle_unit(r_sin_shuf, self.shift_num, self.shuffle_groups, begin=0)
            else:
                x = features[:, 1:]
            
            # lf_1
            b1_local_feat = x[:, :patch_length]
            if r_cos is not None:
                rc_1 = torch.cat([r_cos_f[0:1], r_cos_shuf[0, :patch_length]], dim=0)
                rs_1 = torch.cat([r_sin_f[0:1], r_sin_shuf[0, :patch_length]], dim=0)
            else:
                rc_1, rs_1 = None, None
            b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1), rc_1, rs_1)
            local_feat_1 = b1_local_feat[:, 0]

            # lf_2
            b2_local_feat = x[:, patch_length:patch_length*2]
            if r_cos is not None:
                rc_2 = torch.cat([r_cos_f[0:1], r_cos_shuf[0, patch_length:patch_length*2]], dim=0)
                rs_2 = torch.cat([r_sin_f[0:1], r_sin_shuf[0, patch_length:patch_length*2]], dim=0)
            else:
                rc_2, rs_2 = None, None
            b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1), rc_2, rs_2)
            local_feat_2 = b2_local_feat[:, 0]

            # lf_3
            b3_local_feat = x[:, patch_length*2:patch_length*3]
            if r_cos is not None:
                rc_3 = torch.cat([r_cos_f[0:1], r_cos_shuf[0, patch_length*2:patch_length*3]], dim=0)
                rs_3 = torch.cat([r_sin_f[0:1], r_sin_shuf[0, patch_length*2:patch_length*3]], dim=0)
            else:
                rc_3, rs_3 = None, None
            b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1), rc_3, rs_3)
            local_feat_3 = b3_local_feat[:, 0]

            # lf_4
            b4_local_feat = x[:, patch_length*3:patch_length*4]
            if r_cos is not None:
                rc_4 = torch.cat([r_cos_f[0:1], r_cos_shuf[0, patch_length*3:patch_length*4]], dim=0)
                rs_4 = torch.cat([r_sin_f[0:1], r_sin_shuf[0, patch_length*3:patch_length*4]], dim=0)
            else:
                rc_4, rs_4 = None, None
            b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1), rc_4, rs_4)
            local_feat_4 = b4_local_feat[:, 0]
        # ==============================================================

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num):
    '''
    参数：
        cfg: 配置文件
        num_class: 类别数
        camera_num: 相机数
        view_num: 视角数
    '''
    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model
