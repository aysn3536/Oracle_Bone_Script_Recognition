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

# ����ģ��
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 32 * 32, 76)  # ���辭���ػ����ͼ���СΪ32*32 �����ֵ����ʵ�����������

    def forward(self, x):
        x = torch.relu(self.maxpool(self.conv1(x)))
        x = torch.relu(self.maxpool(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
    
# �����б�ǩ���ݼ�
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

        # �������б�ǩ����self.labels�У����ｫ����ת��������
                    
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
        label = self.label_to_idx[label] # ���ı���ǩת��Ϊ����
        image = self.transform(image)

        return image, label
    
# �����ޱ�ǩ���ݼ�
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
        image = Image.open(image_path).convert('L')  # ����ͼƬ��RGB��ʽ
        image = self.transform(image)
        return image, str(image_path)
    
# ����ת���ܵ�
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
model_name = "cnn_model_7.pth"
cs = 3 # cs = 1 ʱ������ģ��ѵ����
       # cs = 2 ʱ������ģ�����Ԥ������
       # cs = 3 ʱ�����е����ʵ�ʶ������
if cs == 1: # ѵ��ģ��
    
    root_dir = '4_Recognize/ѵ����'

    dataset = CustomDataset(root_dir=root_dir, transform=transform) # ��������

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)# ��������

    model = Net().to(device)  # ʵ����ģ��
    
    criterion = nn.CrossEntropyLoss()  # ��ʧ����
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # �Ż���
    
    epochs = 7 # ��������
    
    if 1 == 2: # ��ʱ����ģ��ѵ������
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device) 

                optimizer.zero_grad()  # �ݶ�����

                outputs = model(inputs)  # ǰ�򴫲�
                loss = criterion(outputs, labels)  # ������ʧ
                loss.backward()  # ���򴫲�
                optimizer.step()  # �Ż�

                running_loss += loss.item()
                if i % 100 == 99:  # ÿ100��batch��ӡһ��ѵ��״̬
                    print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0

        print('Finished Training')

        torch.save(model.state_dict(), model_name)
        
    model_path = model_name
    
    # model = Net().to(device) # ����ģ�ͽṹ
    
    model_state_dict = torch.load(model_path) # ����ģ��
    
    model.load_state_dict(model_state_dict) # ����ģ��
    
    model.eval()  # ��ģ������Ϊ����ģʽ

    true_labels = []
    
    predicted_labels = []

    with torch.no_grad():  # �������ݶȣ��Լӿ�����ٶ�
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
        
            _, predicted = torch.max(outputs.data, 1)
        
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
            
    # ��������ָ��
    precision = precision_score(true_labels, predicted_labels, average='weighted')  # ��׼��
    recall = recall_score(true_labels, predicted_labels, average='weighted')  # ��ȫ��
    f1 = f1_score(true_labels, predicted_labels, average='weighted')  # F1����

    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1 Score: {f1:.3f}')
    
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
if cs == 2: # ʹ��ģ��
    model_path = 'cnn_model_7.pth' # ģ��·��
    model = Net().to(device) # ����ģ�ͽṹ
    model_state_dict = torch.load(model_path) # ����ģ��
    model.load_state_dict(model_state_dict) # ����ģ��
    model.eval() # ģ������Ϊ����ģʽ
    
    folder_path = os.path.join('4_Recognize', 'ѵ����','��') # ���Լ��ļ���
    test_dataset = UnlabeledDataset(root_dir = folder_path, transform=transform) # ���ز��Լ��ļ�
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) # ���ز��Լ��ļ�
    
    root_dir = '4_Recognize/ѵ����'
    dataset = CustomDataset(root_dir=root_dir, transform=transform) # �������ݣ���Ҫ�Ǽ������ֵ����ֵ��ֵ䣩
    
    for images in test_dataloader:
        
        images = images.to(device)  # ת�Ƶ���Ӧ�豸

        if 1 == 2: # �Ƿ�չʾͼƬ
            npimg = images.cpu().numpy()
            plt.imshow(np.transpose(npimg[0], (1, 2, 0)), cmap='gray')
            plt.show()

        with torch.no_grad(): # ����Ԥ��
            output = model(images)

        _, predicted = torch.max(output.data, 1) # ���ģ��Ԥ����
        num_label =  predicted.item()
        chinese_label = dataset.get_label(num_label)
        print("Predicted: ", chinese_label)
        
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
def set_black_to_gray(value):
    # �������ֵΪ��ɫ��0�����򷵻�128�����򱣳�ԭֵ��
    return 128 if value == 0 else value

if cs == 3: # �����Զ�ʶ������
    if 1 == 2: # ������һ��
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
        

    model_path = model_name # ģ��·��
    model = Net().to(device) # ����ģ�ͽṹ
    model_state_dict = torch.load(model_path) # ����ģ��
    model.load_state_dict(model_state_dict) # ����ģ��
    model.eval() # ģ������Ϊ����ģʽ

    folder_path = os.path.join('saved_crops', 'cutted_pics') # ���Լ��ļ���

    test_dataset = UnlabeledDataset(root_dir = folder_path, transform=transform) # ���ز��Լ��ļ�
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) # ���ز��Լ��ļ�

    root_dir = '4_Recognize/ѵ����' # ��Ҫ��
    dataset = CustomDataset(root_dir=root_dir, transform=transform) # �������ݣ���Ҫ�Ǽ������ֵ����ֵ��ֵ䣩���Ҳ��
    
    for images, image_path in test_dataloader:
        the_img_path = image_path[0]
        file_name = os.path.basename(the_img_path)

        images = images.to(device)  # ת�Ƶ���Ӧ�豸
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
        # ��ͼƬ
        with Image.open(img_path) as img:
            img_gray = img.convert('L')

            draw = ImageDraw.Draw(img_gray)

            try:
                # ָ����������ʹ�С
                # �ô���Ҫ�ṩһ��TrueType�����ļ���·��������Ҫ����֧�ֺ��ֵ������ļ�����΢���źڡ�
                font = ImageFont.truetype("msyh.ttc", size=36)
            except IOError:
                print("�����ļ�δ�ҵ�������ʹ��Ĭ�����塣")
                font = ImageFont.load_default()
            
            # ��ָ��������ƺ��֡��á�
            draw.text((int(x1), int(y1)), chinese_label, fill="black", font=font)
        
            # �����޸ĺ��ͼƬ������ѡ�񸲸�ԭͼƬ�򱣴�Ϊ��ͼƬ
            print(img_name)
            img_gray.save(os.path.join('saved_crops','ori_pics', img_name))