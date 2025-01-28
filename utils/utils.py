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