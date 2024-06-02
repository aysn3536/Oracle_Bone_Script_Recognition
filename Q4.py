# -*- coding: gbk -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.io import read_image
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import re
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 32, 76)  # 假设经过池化后的图像大小为32*32 （这个值根据实际情况调整）

    def forward(self, x):
        x = torch.relu(self.maxpool(self.conv1(x)))
        x = torch.relu(self.maxpool(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
# 定义有标签数据集
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, label)
            if os.path.isdir(folder_path):
                for image_name in os.listdir(folder_path):
                    self.images.append(os.path.join(folder_path, image_name))
                    self.labels.append(label)

        # 假设所有标签都在self.labels中，这里将它们转换成数字
                    
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        print("Label to index mapping:", self.label_to_idx)
        print("Index to label mapping:", self.idx_to_label)
        
    def __len__(self):
        return len(self.images)
    
    def get_label(self, idx):
        return self.idx_to_label[idx]
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')
        label = self.labels[idx]
        label = self.label_to_idx[label] # 将文本标签转换为索引
        image = self.transform(image)

        return image, label
    
# 定义无标签数据集
class UnlabeledDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []

        for image_name in os.listdir(root_dir):
            image_path = os.path.join(root_dir, image_name)
            if os.path.isfile(image_path):
                self.images.append(image_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('L')  # 假设图片是RGB格式
        image = self.transform(image)
        return image, str(image_path)
    
# 数据转换管道
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
model_name = "cnn_model_7.pth"
cs = 3 # cs = 1 时，进行模型训练。
       # cs = 2 时，运行模型输出预测结果。
       # cs = 3 时，运行第四问的识别问题
if cs == 1: # 训练模型
    
    root_dir = '4_Recognize/训练集'

    dataset = CustomDataset(root_dir=root_dir, transform=transform) # 加载数据

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)# 加载数据

    model = Net().to(device)  # 实例化模型
    
    criterion = nn.CrossEntropyLoss()  # 损失函数
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 优化器
    
    epochs = 7 # 迭代次数
    
    if 1 == 2: # 暂时跳过模型训练部分
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device) 

                optimizer.zero_grad()  # 梯度清零

                outputs = model(inputs)  # 前向传播
                loss = criterion(outputs, labels)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 优化

                running_loss += loss.item()
                if i % 100 == 99:  # 每100个batch打印一次训练状态
                    print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')

        torch.save(model.state_dict(), model_name)
        
    model_path = model_name
    
    # model = Net().to(device) # 加载模型结构
    
    model_state_dict = torch.load(model_path) # 加载模型
    
    model.load_state_dict(model_state_dict) # 加载模型
    
    model.eval()  # 将模型设置为评估模式

    true_labels = []
    
    predicted_labels = []

    with torch.no_grad():  # 不计算梯度，以加快计算速度
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
        
            _, predicted = torch.max(outputs.data, 1)
        
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            
    # 计算评估指标
    precision = precision_score(true_labels, predicted_labels, average='weighted')  # 查准率
    recall = recall_score(true_labels, predicted_labels, average='weighted')  # 查全率
    f1 = f1_score(true_labels, predicted_labels, average='weighted')  # F1分数

    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
if cs == 2: # 使用模型
    model_path = 'cnn_model_7.pth' # 模型路径
    model = Net().to(device) # 加载模型结构
    model_state_dict = torch.load(model_path) # 加载模型
    model.load_state_dict(model_state_dict) # 加载模型
    model.eval() # 模型设置为评估模式
    
    folder_path = os.path.join('4_Recognize', '训练集','允') # 测试集文件夹
    test_dataset = UnlabeledDataset(root_dir = folder_path, transform=transform) # 加载测试集文件
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) # 加载测试集文件
    
    root_dir = '4_Recognize/训练集'
    dataset = CustomDataset(root_dir=root_dir, transform=transform) # 加载数据（主要是加载数字到汉字的字典）
    
    for images in test_dataloader:
        
        images = images.to(device)  # 转移到相应设备

        if 1 == 2: # 是否展示图片
            npimg = images.cpu().numpy()
            plt.imshow(np.transpose(npimg[0], (1, 2, 0)), cmap='gray')
            plt.show()

        with torch.no_grad(): # 进行预测
            output = model(images)

        _, predicted = torch.max(output.data, 1) # 输出模型预测结果
        num_label =  predicted.item()
        chinese_label = dataset.get_label(num_label)
        print("Predicted: ", chinese_label)
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def set_black_to_gray(value):
    # 如果像素值为黑色（0），则返回128，否则保持原值。
    return 128 if value == 0 else value

if cs == 3: # 文字自动识别任务
    if 1 == 2: # 仅运行一次
        folder = os.path.join('saved_crops','ori_pics2')
        img_names = [img_name for img_name in os.listdir(folder)]

        for img_name in img_names:
            img_path = os.path.join(folder, img_name)
            with Image.open(img_path) as img:
                img_gray = img.convert('L')
                threshold = 128
                img_bin = img_gray.point(lambda p: p > threshold and 255)
                img_gray128 = img_bin.point(set_black_to_gray)
                img_gray128.save(os.path.join('saved_crops','ori_pics', img_name))






    folder = os.path.join('saved_crops','ori_pics')
    pattern1 = r'^[a-zA-Z0-9]+_([0-9]+)_[0-9]+_[0-9]+_[0-9]+\.jpg$'
    pattern2 = r'^[a-zA-Z0-9]+_[0-9]+_([0-9]+)_[0-9]+_[0-9]+\.jpg$'
        

    model_path = model_name # 模型路径
    model = Net().to(device) # 加载模型结构
    model_state_dict = torch.load(model_path) # 加载模型
    model.load_state_dict(model_state_dict) # 加载模型
    model.eval() # 模型设置为评估模式

    folder_path = os.path.join('saved_crops', 'cutted_pics') # 测试集文件夹

    test_dataset = UnlabeledDataset(root_dir = folder_path, transform=transform) # 加载测试集文件
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) # 加载测试集文件

    root_dir = '4_Recognize/训练集' # 不要动
    dataset = CustomDataset(root_dir=root_dir, transform=transform) # 加载数据（主要是加载数字到汉字的字典）这个也别动
    
    for images, image_path in test_dataloader:
        the_img_path = image_path[0]
        file_name = os.path.basename(the_img_path)

        images = images.to(device)  # 转移到相应设备
        with torch.no_grad():
            output = model(images)
        _, predicted = torch.max(output.data, 1)
        num_label =  predicted.item()
        chinese_label = dataset.get_label(num_label)
        print("Predicted: ", chinese_label)



        print(f"filename = {file_name}")
        img_name = file_name[:6] + '.jpg'
        print(img_name)
        img_path = os.path.join(folder,img_name)
        file_name = str(file_name)

        match1 = re.match(pattern1, file_name)
        if match1:
            x1 = match1.group(1)
        match2 = re.match(pattern2, file_name)
        if match2:
            y1 = match2.group(1)
        print(f"y1 = {y1}")
        # 打开图片
        with Image.open(img_path) as img:
            img_gray = img.convert('L')

            draw = ImageDraw.Draw(img_gray)

            try:
                # 指定汉字字体和大小
                # 该处需要提供一个TrueType字体文件的路径。你需要下载支持汉字的字体文件，如微软雅黑。
                font = ImageFont.truetype("msyh.ttc", size=36)
            except IOError:
                print("字体文件未找到，正在使用默认字体。")
                font = ImageFont.load_default()
            
            # 在指定区域绘制汉字“好”
            draw.text((int(x1), int(y1)), chinese_label, fill="black", font=font)
        
            # 保存修改后的图片，可以选择覆盖原图片或保存为新图片
            print(img_name)
            img_gray.save(os.path.join('saved_crops','ori_pics', img_name))