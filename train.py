import torch
import warnings
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

class TrainRuiHui():
    def __init__(self, data_dir, batch_size):
        # GPU, 글씨체 설정(Windows OS)
        if os.name == 'nt' and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("GPU can be available")
            plt.rcParams['font.family'] = 'Malgun Gothic'
        else:
            raise Exception('No CUDA found')
        # (-) 설정
        plt.rcParams['axes.unicode_minus'] = 'False'

        # instances
 
        self.test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
        ])
        self.train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
        ])
        # 폴더 경로를 추가
        sys.path.append(os.path.dirname(os.path.dirname(__file__)))
        self.data_dir = data_dir
        train_dir = os.path.join(self.data_dir, 'train')
        test_dir = os.path.join(self.data_dir, 'val')

        # 훈련용
        self.train_data = datasets.ImageFolder(train_dir, 
                    transform=self.train_transform)
        # 검증용
        self.test_data = datasets.ImageFolder(test_dir, 
                    transform=self.test_transform)    
   
        # 데이터로더 정의
        self.batch_size = batch_size
        # 훈련용
        self.train_loader = DataLoader(self.train_data, 
            batch_size=batch_size, shuffle=True, pin_memory=True)
        # 검증용
        self.test_loader = DataLoader(self.test_data, 
            batch_size=batch_size, shuffle=False, pin_memory=True)
        # 이미지 출력용
        self.test_loader2 = DataLoader(self.test_data, 
            batch_size=50, shuffle=True, pin_memory=True)

        # model
        self.arch = None
        self.net = None      
        self.labels = None  


    def prepare_model(self, arch, label_file_path, weight_path):
        self.arch = arch
        self.labels = np.loadtxt(label_file_path, str, delimiter='\t')
        num_classes = len(self.labels)
        self.net = models.__dict__[self.arch](num_classes)
        self.net = self.net.to(self.device)
        if weight_path != 'default':
            self.net.load_state_dict(torch.load(weight_path, map_location=self.device))


    def fit(self, lr=0.001, epochs=10):
        self.lr = lr  
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(),lr=lr)
        self.epoch = epochs
        self.history = np.zeros((0, 5))
        base_epochs = len(self.history)
    
        for epoch in range(base_epochs, self.epoch+base_epochs):
            train_loss = 0
            train_acc = 0
            val_loss = 0
            val_acc = 0

            # 훈련 페이즈
            self.net.train()
            count = 0

            for inputs, labels in tqdm(self.train_loader):
                count += len(labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 경사 초기화
                self.optimizer.zero_grad()

                # 예측 계산
                outputs = self.net(inputs)

                # 손실 계산
                loss = self.criterion(outputs, labels)
                train_loss += loss.item()

                # 경사 계산
                loss.backward()

                # 파라미터 수정
                self.optimizer.step()

                # 예측 라벨 산출
                predicted = torch.max(outputs, 1)[1]

                # 정답 건수 산출
                train_acc += (predicted == labels).sum().item()

                # 손실과 정확도 계산
                avg_train_loss = train_loss / count
                avg_train_acc = train_acc / count

            # 예측 페이즈
            self.net.eval()
            count = 0

            for inputs, labels in self.test_loader:
                count += len(labels)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # 예측 계산
                outputs = self.net(inputs)

                # 손실 계산
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()

                # 예측 라벨 산출
                predicted = torch.max(outputs, 1)[1]

                # 정답 건수 산출
                val_acc += (predicted == labels).sum().item()

                # 손실과 정확도 계산
                avg_val_loss = val_loss / count
                avg_val_acc = val_acc / count
        
            print (f'Epoch [{(epoch+1)}/{self.epoch+base_epochs}], loss: {avg_train_loss:.5f} acc: {avg_train_acc:.5f} val_loss: {avg_val_loss:.5f}, val_acc: {avg_val_acc:.5f}')
            item = np.array([epoch+1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc])
            self.history = np.vstack((self.history, item))
        return self.history
    

    def evaluate_history(self):
        # Check Loss and Accuracy
        print(f'Initial state: Loss: {self.history[0,3]:.5f}  Accuracy: {self.history[0,4]:.4f}') 
        print(f'Final state: Loss: {self.history[-1,3]:.5f}  Accuracy: {self.history[-1,4]:.4f}')

        num_epochs = len(self.history)
        unit = num_epochs / 10

        # Plot the training curve (Loss)
        plt.figure(figsize=(9,8))
        plt.plot(self.history[:,0], self.history[:,1], 'b', label='Training')
        plt.plot(self.history[:,0], self.history[:,3], 'k', label='Validation')
        plt.xticks(np.arange(0, num_epochs+1, unit))
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Curve (Loss)')
        plt.legend()
        plt.show()

        # Plot the training curve (Accuracy)
        plt.figure(figsize=(9,8))
        plt.plot(self.history[:,0], self.history[:,2], 'b', label='Training')
        plt.plot(self.history[:,0], self.history[:,4], 'k', label='Validation')
        plt.xticks(np.arange(0, num_epochs+1, unit))
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Curve (Accuracy)')
        plt.legend()
        plt.show()


    def show_images_labels(self):
        """
        Show images and their labels from the DataLoader.

        Args:
            loader: PyTorch DataLoader object.
            classes: List of class names.
            net: (Optional) Model for prediction. Default is None.
            device: Device to perform computations on.
        """
        # 데이터로더에서 처음 1세트를 가져오기
        for images, labels in self.test_loader2:
            break

        # 표시할 이미지 개수
        n_size = min(len(images), 50)

        if self.net is not None:
            # 디바이스 할당
            inputs = images.to(self.device)
            labels = labels.to(self.device)

            # 예측 계산
            outputs = self.net(inputs)
            predicted = torch.max(outputs, 1)[1]

        # 플롯 크기 설정
        plt.figure(figsize=(20, 15))
        
        # 처음 n_size개의 이미지 표시
        for i in range(n_size):
            ax = plt.subplot(5, 10, i + 1)  # 5행 10열 서브플롯
            label_name = self.labels[labels[i]]
            
            # self.net이 None이 아닌 경우 예측 결과도 표시
            if self.net is not None:
                predicted_name = self.labels[predicted[i]]
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
    t1 = TrainRuiHui(args.datapath, args.batch_size)
    t1.prepare_model(args.arch,'label_map.txt', args.load_weights)
    print(args)
    utils.torch_seed()
    start_time = time.time()
    t1.fit(args.lr, args.epochs)
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))    
    utils.save_weight(net=t1.net, path=args.save)
    t1.evaluate_history()
    utils.torch_seed()
    t1.show_images_labels()