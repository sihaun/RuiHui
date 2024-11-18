# RuiHui
To distinguish Ruibao and Huibao.

Distinguishing them is so difficult.
When the VGG-19-BN model was used, the accurancy was about 55%.

Therefore, model efficientnet_b7 was used to distinguish the small features of the image.
This currently shows an accuracy of about 85%.

# Default Setting
transfer learning
net = models.efficientnet_b7(pretrained = True)
batch_size = 5
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=lr)
num_epochs = 30

You can use update version of torchlib.py in
https://github.com/sihaun/dlftn.git