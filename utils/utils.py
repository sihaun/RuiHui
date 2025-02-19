import torch


# 모델 가중치 저장
def save_weight(net, path='weight.pt'):
   torch.save(net.state_dict(), path)

def save_model(net, path='model.pt'):
   torch.save(net, path)

# 모델 가중치 불러오기
def load_weights(net, path='weight.pth'):
   net.load_state_dict(torch.load(path))

# 모델 가중치 확인
def show_weights(net):
    for name, param in net.state_dict().items():
        print(f"Layer: {name} | Shape: {param.shape}")
        print(param)


# 파이토치 난수 고정
def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
padding
kernel=1 => padding=0
kernel=3 => padding=1
kernel=5 => padding=2

stride
stride=1 => upsample
stride=2 => downsample

depthwise conv => (group=in_channels)
nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)

pointwise conv => (kerbel_size=1), stride=1, padding=0
nn.Conv2d(in_channels, out_channels, kernel_size=1)

=> Inverted Residual Block
pointwise -> depthwise -> pointwise -> se
input, output, kernel
expand, stride, se, activation

            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
'''
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, expand_channels, out_channels, kernel_size, stride, activation, se_reduction=1):
        self.pointwise1 = PointwiseConv2d(in_channels, expand_channels, activation)
        self.depthwise = DepthwiseConv2d(expand_channels, expand_channels, kernel_size, stride, activation)
        self.pointwise2 = PointwiseConv2d(expand_channels, out_channels, activation)
        if se_reduction > 1:
            self.se = SEBlock(out_channels, se_reduction)
        elif se_reduction < 1:
            torch._assert(False, f"SE Reduction {se_reduction} must be greater than or equal to 1.")

    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.pointwise2(x)
        if self.se:
            x = self.se(x)
        return x
    
class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(DepthwiseConv2d, self).__init__()
        self.conv = Conv2dNormActivation(in_channels, out_channels, kernel_size=kernel_size, stride=1, groups=in_channels, **kwargs)

    def forward(self, x):
        return self.conv(x)
    
class PointwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(PointwiseConv2d, self).__init__()
        self.conv = Conv2dNormActivation(in_channels, out_channels, kernel_size=1, **kwargs)

    def forward(self, x):
        return self.conv(x)


class Conv2dNormActivation(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, groups=1, activation='ReLU', **kwargs):
        super(Conv2dNormActivation, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=groups, **kwargs)
        self.norm = nn.BatchNorm2d(output_channels)
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Hardswish':
            self.activation = nn.Hardswish()
        else:
            torch._assert(False, f"Activation function {activation} is not supported.")
        self.activation = activation


    def forward(self, x):
        return x
    
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # Channel reduction을 위한 첫 번째 FC layer
        self.fc1 = nn.Linear(channel, channel // reduction)
        # 복원되는 FC layer
        self.fc2 = nn.Linear(channel // reduction, channel)
        # Sigmoid 활성화 함수 (채널 중요도 계산)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling (GAP)
        b, c, _, _ = x.size()  # b: batch size, c: channels
        gap = torch.mean(x, dim=(2, 3), keepdim=False)  # (B, C) 형태로 GAP을 계산

        # Fully Connected layer를 통해 중요도 학습
        gap = self.fc1(gap)  # (B, C//reduction)
        gap = torch.relu(gap)  # ReLU 활성화
        gap = self.fc2(gap)  # (B, C)

        # Sigmoid를 통해 0과 1 사이의 값으로 중요도 조절
        gap = self.sigmoid(gap).view(b, c, 1, 1)  # (B, C, 1, 1)
        
        # 입력 특성 맵에 채널별 중요도를 곱하여 가중치를 부여
        return x * gap  # (B, C, H, W) 형태로 스케일링된 출력
    

class MultiBoxLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, neg_pos_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.alpha = alpha  # Localization과 Classification Loss의 가중치
        self.neg_pos_ratio = neg_pos_ratio  # Hard Negative Mining 비율

    def forward(self, confidences, locations, gt_labels, gt_locations):
        """
        confidences: (batch, num_anchors, num_classes)  # 분류 결과
        locations: (batch, num_anchors, 4)  # 박스 좌표 (cx, cy, w, h)
        gt_labels: (batch, num_anchors)  # Ground Truth 라벨
        gt_locations: (batch, num_anchors, 4)  # Ground Truth 박스
        """

        # Localization Loss (Smooth L1)
        pos_mask = gt_labels > 0  # Positive (객체가 있는) 앵커 마스크
        loc_loss = F.smooth_l1_loss(locations[pos_mask], gt_locations[pos_mask], reduction='sum')

        # Confidence Loss (Cross Entropy)
        conf_loss = F.cross_entropy(confidences.view(-1, confidences.size(-1)), gt_labels.view(-1), reduction='none')
        conf_loss = conf_loss.view(confidences.shape[:2])

        # Hard Negative Mining
        neg_mask = gt_labels == 0  # Negative (배경) 앵커 마스크
        num_pos = pos_mask.sum(dim=1, keepdim=True)  # Positive 샘플 개수
        num_neg = self.neg_pos_ratio * num_pos  # Negative 샘플 개수 제한
        num_neg = torch.clamp(num_neg, max=neg_mask.sum(dim=1, keepdim=True))  # 최대 개수 제한

        # Negative 샘플 중 loss가 큰 것만 선택
        conf_loss_neg = conf_loss.clone()
        conf_loss_neg[pos_mask] = 0  # Positive 샘플 제거
        _, neg_idx = conf_loss_neg.sort(dim=1, descending=True)
        neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
        for i in range(neg_mask.size(0)):
            neg_mask[i, neg_idx[i, :num_neg[i].item()]] = True

        # 최종 Confidence Loss
        conf_loss = conf_loss[pos_mask | neg_mask].sum()

        # 최종 Loss 계산
        total_loss = (loc_loss + self.alpha * conf_loss) / num_pos.sum()
        return total_loss
