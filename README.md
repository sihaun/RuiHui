# RuiHui
To distinguish the pandas including Ruibao and Huibao.

Pandas that can be distinguished : Aibao, Fubao, Huibao, Lebao, Ruibao.

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

If you want to fix configuration, fix config.json or train_bao.sh
