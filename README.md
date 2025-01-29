# RuiHui
To distinguish the pandas including Ruibao and Huibao.

Pandas that can be distinguished : Aibao, Fubao, Huibao, Lebao, Ruibao.

## Models
Currently, all training and reference courses use the model **EfficientNet_b7**. I used EfficientNet's largest model (except V2) and **Transfer Learning** to achieve great effect even with a small dataset. The EfficientNet_b7 model can show meaningful classification results even with small differences. The default weight of the model is **DEFAULT** on **efficientnet_b7**.
```Python
  for param in self.net.parameters():
      param.requires_grad = False
  
  in_features = self.net.classifier[1].in_features
  self.net.classifier[1] = nn.Linear(in_features, num_classes)
```
In addition, this model **does not contain Softmax in classifier** like EfficientNet_b7.
```Python
  def forward(self, x):
      x = self.net(x)
      return x
```

## Train
Thankfully for the large model, even a small dataset resulted in meaningful accuracy training results. This currently shows an accuracy of **about 85%**. However, it is **not recommended** for practical use because it is still difficult to use in practice.

For learning, you can run the code like this:
```bash
python train.py --arch tf_efficientnet_b7 --datapath image_data --epochs 3 --batch-size 5 --lr 0.001 --save weight.pt
```
If you want to change the image label differently, modify the **label_map.txt**
![Figure_1](https://github.com/user-attachments/assets/db1228d3-3df6-44b9-8848-5513e4d59c37)
![Figure_2](https://github.com/user-attachments/assets/38414138-8463-4d13-9951-f56ba41a480d)
![Figure_3](https://github.com/user-attachments/assets/77d4c73a-672e-411f-8560-7d39db05ac5c)
![figure_4](https://github.com/user-attachments/assets/3f82ec5d-9708-44cb-96eb-d016709bdaa5)

### Default Setting

| Type | Value | Note |
|------|-------|-------------|
| Learning Type | transfer learning |  |
| Model | tf_efficientnet_b7 | pretrained = True |
| Batch Size | 5 |     |
| Learning Rate | 0.001 |  |
| Criterion | CrossEntropyLoss |  |
| Optimizer | Adam |  |
| Epochs | 30 |  |

## InferFromEffinet
Image or video inference can be performed with the weight that has completed learning. You can choose a mode to infer **images or videos**.

If you want to infer images, use the command in the following format
```bash
python InferFromEffinet.py --mode Image --weight weight.pt --source test_image.jpg
```
If you want to infer videos, use the command in the following format
```bash
python InferFromEffinet.py --mode Video --weight weight.pt --source test_video.mp4
```




