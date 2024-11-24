from torchlib import *
from distribute_images import *
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import sys

# 폴더 경로를 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 분류하려는 클래스의 리스트 작성
classes = ['aibao','fubao','huibao','lebao','ruibao']


src_dir = 'all_image'
data_dir = 'image_data'

distribute_images(source_dir=src_dir, output_dir=data_dir, classes=classes, train_ratio=0.8, rename=True)

# Transforms 정의

# 검증 데이터 : 정규화
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

# 훈련 데이터 : 정규화에 반전과 RandomErasing 추가
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
])

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')

# 데이터셋 정의

# 훈련용
train_data = datasets.ImageFolder(train_dir, 
            transform=train_transform)
# 훈련 데이터 이미지 출력용
train_data2 = datasets.ImageFolder(train_dir, 
            transform=test_transform)
# 검증용
test_data = datasets.ImageFolder(test_dir, 
            transform=test_transform)

# 데이터로더 정의

batch_size = 5

# 훈련용
train_loader = DataLoader(train_data, 
      batch_size=batch_size, shuffle=True)

# 검증용
test_loader = DataLoader(test_data, 
      batch_size=batch_size, shuffle=False)

# 이미지 출력용
train_loader2 = DataLoader(train_data2, 
      batch_size=50, shuffle=True)
test_loader2 = DataLoader(test_data, 
      batch_size=50, shuffle=True)

# 전이 학습의 경우

# efficientnet_b7 모델을 학습이 끝난 파라미터와 함께 불러오기
from torchvision import models
net = models.efficientnet_b7(pretrained = True)

# 모든 파라미터의 경사 계산을 OFF로 설정
for param in net.parameters():
    param.requires_grad = False

# 난수 고정
torch_seed()

# EfficientNet
# 최종 노드의 출력을 2로 변경
# 이 노드에 대해서만 경사 계산을 수행하게 됨
# 최종 Fully Connected Layer를 새롭게 정의
num_classes = len(classes)
in_features = net.classifier[1].in_features  # 기존 클래스 개수의 입력 차원 가져오기
net.classifier[1] = nn.Linear(in_features, num_classes)

# GPU 사용
net = net.to(device)

# 학습률
lr = 0.001

# 손실 함수로 교차 엔트로피 사용
criterion = nn.CrossEntropyLoss()

# 최적화 함수 정의
# 파라미터 수정 대상을 최종 노드로 제한
optimizer = optim.Adam(net.parameters(),lr=lr)

# history 파일도 동시에 초기화
history = np.zeros((0, 5))

# 학습
num_epochs = 30
history = fit(net, optimizer, criterion, num_epochs, 
          train_loader, test_loader, device, history)

save_weights(net=net)

# 결과 확인
evaluate_history(history)

# 난수 고정
torch_seed()

# 검증 데이터 결과 출력
show_images_labels(test_loader2, classes, net, device)
