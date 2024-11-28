# RuiHui
To distinguish the pandas including Ruibao and Huibao.

Pandas that can be distinguished : Aibao, Fubao, Huibao, Lebao, Ruibao.

When the VGG-19-BN model was used, the accurancy was about 55%.

Therefore, model efficientnet_b7 was used to distinguish the small features of the image.
This currently shows an accuracy of about 85%.
![Figure_1](https://github.com/user-attachments/assets/22545d7b-16b8-4d61-8914-1af26c048cea)
![Figure_2](https://github.com/user-attachments/assets/10a3bf60-1690-42d5-b748-9bf4563c8138)
![Figure_3](https://github.com/user-attachments/assets/77d4c73a-672e-411f-8560-7d39db05ac5c)
![figure_4](https://github.com/user-attachments/assets/98c16f23-d556-429c-a77c-05ec3bdc9664)

# Default Setting
transfer learning

net = models.efficientnet_b7(pretrained = True)

batch_size = 5

lr = 0.001

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(),lr=lr)

num_epochs = 30

If you want to fix configuration, fix config.json or train_bao.sh
