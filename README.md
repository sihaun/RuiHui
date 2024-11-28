# RuiHui
To distinguish the pandas including Ruibao and Huibao.

Pandas that can be distinguished : Aibao, Fubao, Huibao, Lebao, Ruibao.

When the VGG-19-BN model was used, the accurancy was about 55%.

Therefore, model **efficientnet_b7** was used to distinguish the small features of the image.
This currently shows an accuracy of about 85%.

![Figure_1](https://github.com/user-attachments/assets/db1228d3-3df6-44b9-8848-5513e4d59c37)
![Figure_2](https://github.com/user-attachments/assets/38414138-8463-4d13-9951-f56ba41a480d)
![Figure_3](https://github.com/user-attachments/assets/77d4c73a-672e-411f-8560-7d39db05ac5c)
![figure_4](https://github.com/user-attachments/assets/3f82ec5d-9708-44cb-96eb-d016709bdaa5)


# Default Setting
transfer learning

net = models.efficientnet_b7(pretrained = True)

batch_size = 5

lr = 0.001

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(),lr=lr)

num_epochs = 30

**If you want to fix configuration, fix config.json or train_bao.sh**
