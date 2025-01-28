# RuiHui
To distinguish the pandas including Ruibao and Huibao.

Pandas that can be distinguished : Aibao, Fubao, Huibao, Lebao, Ruibao.

Model **efficientnet_b7** was used to distinguish the small features of the image.
This currently shows an accuracy of about 85%.

![Figure_1](https://github.com/user-attachments/assets/db1228d3-3df6-44b9-8848-5513e4d59c37)
![Figure_2](https://github.com/user-attachments/assets/38414138-8463-4d13-9951-f56ba41a480d)
![Figure_3](https://github.com/user-attachments/assets/77d4c73a-672e-411f-8560-7d39db05ac5c)
![figure_4](https://github.com/user-attachments/assets/3f82ec5d-9708-44cb-96eb-d016709bdaa5)

# Default Setting

| Type | Value | Note |
|------|-------|-------------|
| Learning Type | transfer learning |  |
| Model | tf_efficientnet_b7 | pretrained = True |
| Batch Size | 5 |     |
| Learning Rate | 0.001 |  |
| Criterion | CrossEntropyLoss |  |
| Optimizer | Adam |  |
| Epochs | 30 |  |


