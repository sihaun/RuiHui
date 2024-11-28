import torch
import warnings
from distribute_images import *
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys
import config
import time
import utils
import models

warnings.simplefilter('ignore')

def hyperparam():
    args = config.config()
    return args

def main(args):

    # GPU, 글씨체 설정(Mac OS)
    if os.name == 'posix' and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("GPU can be available")
        plt.rcParams['font.family'] = 'AppleGothic'
    # GPU, 글씨체 설정(Windows OS)
    elif os.name == 'nt' and torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("GPU can be available")
        plt.rcParams['font.family'] = 'Malgun Gothic'
    else:
        raise Exception('No CUDA found')
    # (-) 설정
    plt.rcParams['axes.unicode_minus'] = 'False'
    
    if not args.cuda:
        raise Exception('No GPU found, please run without --cuda')
        
    # 분류하려는 클래스
    classes = args.classes

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

    # 폴더 경로를 추가
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    data_dir = args.datapath
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'val')

    # 데이터셋 정의

    # 훈련용
    train_data = datasets.ImageFolder(train_dir, 
                transform=train_transform)
    # 검증용
    test_data = datasets.ImageFolder(test_dir, 
                transform=test_transform)

    # 데이터로더 정의

    batch_size = args.batch_size

    # 훈련용
    train_loader = DataLoader(train_data, 
        batch_size=batch_size, shuffle=True)

    # 검증용
    test_loader = DataLoader(test_data, 
        batch_size=batch_size, shuffle=False)

    # 이미지 출력용
    test_loader2 = DataLoader(test_data, 
        batch_size=50, shuffle=True)

    # 전이 학습의 경우
    net = models.__dict__[args.arch](num_classes = len(classes))

    # 난수 고정
    utils.torch_seed()

    # GPU 사용
    net = net.to(device)

    # 학습률
    lr = args.lr

    # 손실 함수로 교차 엔트로피 사용
    criterion = nn.CrossEntropyLoss()

    # 최적화 함수 정의
    # 파라미터 수정 대상을 최종 노드로 제한
    optimizer = optim.Adam(net.parameters(),lr=lr)

    # history 파일도 동시에 초기화
    history = np.zeros((0, 5))

    # 학습
    num_epochs = args.epochs
    start_time = time.time()

    history = fit(net, optimizer, criterion, num_epochs, 
        train_loader, test_loader, device, history)
    
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))
    
    utils.save_weights(net=net, path=args.save)

    # 결과 확인
    evaluate_history(history)

    # 난수 고정
    utils.torch_seed()

    # 검증 데이터 결과 출력
    show_images_labels(test_loader2, classes, net, device)


def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
    
    base_epochs = len(history)
  
    for epoch in range(base_epochs, num_epochs+base_epochs):
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        # 훈련 페이즈
        net.train()
        count = 0

        for inputs, labels in tqdm(train_loader):
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 경사 초기화
            optimizer.zero_grad()

            # 예측 계산
            outputs = net(inputs)

            # 손실 계산
            loss = criterion(outputs, labels)
            train_loss += loss.item()

            # 경사 계산
            loss.backward()

            # 파라미터 수정
            optimizer.step()

            # 예측 라벨 산출
            predicted = torch.max(outputs, 1)[1]

            # 정답 건수 산출
            train_acc += (predicted == labels).sum().item()

            # 손실과 정확도 계산
            avg_train_loss = train_loss / count
            avg_train_acc = train_acc / count

        # 예측 페이즈
        net.eval()
        count = 0

        for inputs, labels in test_loader:
            count += len(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 예측 계산
            outputs = net(inputs)

            # 손실 계산
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 예측 라벨 산출
            predicted = torch.max(outputs, 1)[1]

            # 정답 건수 산출
            val_acc += (predicted == labels).sum().item()

            # 손실과 정확도 계산
            avg_val_loss = val_loss / count
            avg_val_acc = val_acc / count
    
        print (f'Epoch [{(epoch+1)}/{num_epochs+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
        item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
        history = np.vstack((history, item))
    return history

# 손실 계산용
def eval_loss(loader, device, net, criterion):
  
    # 데이터로더에서 처음 한 개 세트를 가져옴
    for images, labels in loader:
        break

    # 디바이스 할당
    inputs = images.to(device)
    labels = labels.to(device)

    # 예측 계산
    outputs = net(inputs)

    # 손실 계산
    loss = criterion(outputs, labels)

    return loss

def evaluate_history(history):
    # 손실과 정확도 확인
    print(f'초기상태 : 손실 : {history[0,3]:.5f}  정확도 : {history[0,4]:.4f}') 
    print(f'최종상태 : 손실 : {history[-1,3]:.5f}  정확도 : {history[-1,4]:.4f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 학습 곡선 출력(손실)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='훈련')
    plt.plot(history[:,0], history[:,3], 'k', label='검증')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('반복 횟수')
    plt.ylabel('손실')
    plt.title('학습 곡선(손실)')
    plt.legend()
    plt.show()

    # 학습 곡선 출력(정확도)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='훈련')
    plt.plot(history[:,0], history[:,4], 'k', label='검증')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('반복 횟수')
    plt.ylabel('정확도')
    plt.title('학습 곡선(정확도)')
    plt.legend()
    plt.show()

# 이미지와 라벨 표시
def show_images_labels(loader, classes, net=None, device='cpu'):
    """
    Show images and their labels from the DataLoader.

    Args:
        loader: PyTorch DataLoader object.
        classes: List of class names.
        net: (Optional) Model for prediction. Default is None.
        device: Device to perform computations on.
    """
    # 데이터로더에서 처음 1세트를 가져오기
    for images, labels in loader:
        break

    # 표시할 이미지 개수
    n_size = min(len(images), 50)

    if net is not None:
        # 디바이스 할당
        inputs = images.to(device)
        labels = labels.to(device)

        # 예측 계산
        outputs = net(inputs)
        predicted = torch.max(outputs, 1)[1]

    # 플롯 크기 설정
    plt.figure(figsize=(20, 15))
    
    # 처음 n_size개의 이미지 표시
    for i in range(n_size):
        ax = plt.subplot(5, 10, i + 1)  # 5행 10열 서브플롯
        label_name = classes[labels[i]]
        
        # net이 None이 아닌 경우 예측 결과도 표시
        if net is not None:
            predicted_name = classes[predicted[i]]
            # 정답 여부에 따라 색상 설정
            color = 'k' if label_name == predicted_name else 'b'
            ax.set_title(f'{label_name}\n{predicted_name}', c=color, fontsize=14)
        else:
            ax.set_title(label_name, fontsize=14)
        
        # 텐서를 넘파이로 변환
        image_np = images[i].numpy().copy()
        # (channel, row, column) -> (row, column, channel)로 변경
        img = np.transpose(image_np, (1, 2, 0))
        # 값의 범위를 [-1, 1] -> [0, 1]로 조정
        img = (img + 1) / 2
        
        # 이미지 표시
        plt.imshow(img)
        ax.set_axis_off()

    # 설명 추가
    plt.subplots_adjust(top=0.9)  # 제목과 플롯 간격 조정
    plt.suptitle("Black: Correct, Blue: Incorrect", fontsize=18, color='gray')
    plt.show()

if __name__ == "__main__":
    args = hyperparam()
    print(args)
    main(args)