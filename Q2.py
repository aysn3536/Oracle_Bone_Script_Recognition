# -*- coding: gbk -*-
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import json
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2


'''
class UNet(nn.Module): # 旧模型
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器
        self.enc_conv1 = self.encoder_block(1, 8)
        self.enc_conv2 = self.encoder_block(8, 16)
        self.enc_conv3 = self.encoder_block(16, 32)
        self.enc_conv4 = self.encoder_block(32, 128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 解码器
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv4 = self.decoder_block(192, 64)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv3 = self.decoder_block(64, 32)

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv2 = self.decoder_block(32, 16)
        
        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec_conv1 = self.decoder_block(16, 8)
        # 最终层
        self.final_conv = nn.Conv2d(8, 1, kernel_size=1)


    def encoder_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        return block
    def decoder_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        return block
    def forward(self, x):
        # 编码
        enc1 = self.enc_conv1(x)
        pool1 = self.pool(enc1)
        
        enc2 = self.enc_conv2(pool1)
        pool2 = self.pool(enc2)
        
        enc3 = self.enc_conv3(pool2)
        pool3 = self.pool(enc3)
        
        enc4 = self.enc_conv4(pool3)
        pool4 = self.pool(enc4)
        
        # 解码
        up4 = self.upconv4(pool4)
        merge4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec_conv4(merge4)
        
        up3 = self.upconv3(dec4)
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec_conv3(merge3)
        
        up2 = self.upconv2(dec3)
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec_conv2(merge2)
        
        up1 = self.upconv1(dec2)
        merge1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec_conv1(merge1)
        # 最终层
        return torch.sigmoid(self.final_conv(dec1))
# 函数，寻找最大内接矩形
def find_max_inner_rectangle(contour, orig_image_shape):
    x_min, y_min, x_max, y_max = cv2.boundingRect(contour)
    max_area = 0
    rect_coords = (x_min, y_min, x_min, y_min)  # Default rect_coords
    
    # Try each point as potential bottom-right corner
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            top_left = (x_min, y_min)
            bottom_right = (x, y)
            test_rect = (top_left[0], top_left[1], bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
            test_area = test_rect[2] * test_rect[3]
            if test_area > max_area and cv2.pointPolygonTest(contour, bottom_right, False) >= 0:
                max_area = test_area
                rect_coords = test_rect
    return rect_coords
''' # 旧的模型
def get_json_files_in_folder(folder_path): # 获取这个文件夹下的所有json文件，以列表形式输出
    json_files = []
    files = os.listdir(folder_path)
    i = 1
    for file in files:
        if file.endswith('.json'):
            i += 1
            json_files.append(os.path.join(folder_path, file))
        if i > picture_num:
            break
    return json_files

def calculate_iou(preds, labels):
    # 计算交集
    intersection = (preds & labels).float().sum((1, 2))
    # 计算并集
    union = (preds | labels).float().sum((1, 2))
    iou = intersection / (union + 1e-6)
    return iou.mean()  # 返回批次平均IoU

def calculate_dice(preds, labels):
    # 计算Dice系数
    intersection = (preds & labels).float().sum((1, 2))
    dice = (2 * intersection) / (preds.float().sum((1, 2)) + labels.float().sum((1, 2)) + 1e-6)
    return dice.mean()  # 返回批次平均Dice系数

def calculate_pixel_accuracy(preds, labels):
    # 计算像素准确度
    correct = (preds == labels).float().sum((1, 2))
    total = labels.numel() / labels.size(0)  # 每张图像的像素总数
    return correct / total  # 返回批次平均Pixel Accuracy

class AttentionGate(nn.Module): # 注意力机制
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 编码
        enc1 = self.enc_conv1(x)
        pool1 = self.pool(enc1)
    
        enc2 = self.enc_conv2(pool1)
        pool2 = self.pool(enc2)
    
        enc3 = self.enc_conv3(pool2)
        pool3 = self.pool(enc3)
    
        enc4 = self.enc_conv4(pool3)
        pool4 = self.pool(enc4)
    
        # 解码
        up4 = self.upconv4(pool4)
        # 注意力机制集成前的merge改为使用注意力门后的融合
        att4 = AttentionGate(F_g=64, F_l=128, F_int=64)(up4, enc4)
        merge4 = torch.cat([up4, att4], dim=1)
        dec4 = self.dec_conv4(merge4)
    
        up3 = self.upconv3(dec4)
        att3 = AttentionGate(F_g=32, F_l=64, F_int=32)(up3, enc3)
        merge3 = torch.cat([up3, att3], dim=1)
        dec3 = self.dec_conv3(merge3)
    
        up2 = self.upconv2(dec3)
        att2 = AttentionGate(F_g=16, F_l=32, F_int=16)(up2, enc2)
        merge2 = torch.cat([up2, att2], dim=1)
        dec2 = self.dec_conv2(merge2)
    
        up1 = self.upconv1(dec2)
        att1 = AttentionGate(F_g=8, F_l=16, F_int=8)(up1, enc1)
        merge1 = torch.cat([up1, att1], dim=1)
        dec1 = self.dec_conv1(merge1)
        # 最终层
        return torch.sigmoid(self.final_conv(dec1))

class UNet(nn.Module): # 核心的机器学习部分
    def __init__(self):
        super(UNet, self).__init__()
        # 编码器
        self.enc_conv1 = self.encoder_block(1, 8)
        self.enc_conv2 = self.encoder_block(8, 16)
        self.enc_conv3 = self.encoder_block(16, 32)
        self.enc_conv4 = self.encoder_block(32, 128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 解码器
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv4 = self.decoder_block(192, 64)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv3 = self.decoder_block(64, 32)

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv2 = self.decoder_block(32, 16)
        
        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec_conv1 = self.decoder_block(16, 8)
        # 最终层
        self.final_conv = nn.Conv2d(8, 1, kernel_size=1)

    def encoder_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        return block
    def decoder_block(self, in_channels, out_channels):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        return block
    def forward(self, x):
        # 编码
        enc1 = self.enc_conv1(x)
        pool1 = self.pool(enc1)
        
        enc2 = self.enc_conv2(pool1)
        pool2 = self.pool(enc2)
        
        enc3 = self.enc_conv3(pool2)
        pool3 = self.pool(enc3)
        
        enc4 = self.enc_conv4(pool3)
        pool4 = self.pool(enc4)
        
        # 解码
        up4 = self.upconv4(pool4)
        merge4 = torch.cat([up4, enc4], dim=1)
        dec4 = self.dec_conv4(merge4)
        
        up3 = self.upconv3(dec4)
        merge3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec_conv3(merge3)
        
        up2 = self.upconv2(dec3)
        merge2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec_conv2(merge2)
        
        up1 = self.upconv1(dec2)
        merge1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec_conv1(merge1)
        # 最终层
        return torch.sigmoid(self.final_conv(dec1))

class OracleBoneDataset(Dataset): # 定义数据集
    def __init__(self, image_dir, json_files):
        self.image_dir = image_dir
        self.json_files = json_files
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        with open(json_file) as f:
            ann = json.load(f)
        img_name = ann['img_name'] + '.jpg'
        img_ann = ann['ann']
        # print(type(img_ann))
        image_path = os.path.join(self.image_dir, img_name)
        
        # image_path = os.path.join(self.image_dir, ann['img_name'] + '.jpg')
        image = Image.open(image_path).convert('L')
        
        image_x = image.resize((128, 128))
        
        image = self.transform(image)
        image_x = self.transform(image_x)
        mask = np.zeros((image.shape[1], image.shape[2]), dtype=np.float32)
        for region in ann['ann']:
            # visualize_image_with_region(image_x, region)
            x1, y1, x2, y2, _ = region
            mask[int(y1):int(y2), int(x1):int(x2)] = 1.0
            
        mask = torch.tensor(mask).unsqueeze(0)
        # plt.imshow(mask.squeeze(0), cmap='gray')
        # plt.title('Mask')
        # plt.show()
        
        new_mask = np.zeros((128, 128), dtype=np.float32)

        # 将原始mask调整为128x128像素大小
        resized_mask = np.array(Image.fromarray(mask.squeeze(0).numpy()).resize((128, 128)))

        # 将调整大小后的mask放入新的128x128数组中
        new_mask[:resized_mask.shape[0], :resized_mask.shape[1]] = resized_mask

        # 将新的mask转换为PyTorch张量
        new_mask = torch.tensor(new_mask).unsqueeze(0)
        threshold = 0.5
        new_mask[new_mask < threshold] = 0
        new_mask[new_mask >= threshold] = 1

        # 可视化阈值化后的mask
        # plt.imshow(new_mask.squeeze(0), cmap='gray')
        # plt.title('Binary Mask')
        # plt.show()

        return image_x, new_mask, img_name, img_ann

picture_num = 6148 # 生成模型时，总共处理的图片数量
num_epochs = 25 # 生成模型时，迭代次数
outputs_threshold = 0.5 # 运行模型时的输出阈值从这个值开始，以0.05递减
model_path = 'unet_model_6148_25_att.pth' # 选择模型保存/使用的路径

cs = 2 # 1, 训练模型
       # 2, 评估模型



if cs == 1: # 训练模型
    
    image_dir = '2_Train' # 实例化数据集 需要替换成自己的路径
    # json_files = ['2_Train_bf/b02519Z.json', '2_Train_bf/b02523F.json']
    json_files = get_json_files_in_folder(image_dir)
    dataset = OracleBoneDataset(image_dir, json_files)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False) # 数据加载器
    print("数据加载完成")
    model = UNet() # 实例化模型
    print("模型实例化完成")
    criterion = nn.BCELoss() # 损失函数
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # 优化器
    print("损失函数和优化器构建完成")
    # 训练模型
    for epoch in range(num_epochs):
        print(f"开始第{epoch + 1}/{num_epochs}次训练模型")
        model.train()
        for images, masks, _, _  in data_loader:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), model_path) # 保存模型权重
    print("Model training complete and saved.")
    
while outputs_threshold > 0.001 and cs == 2: # 评估模型
    model_state_dict = torch.load(model_path)

    # 实例化模型
    model = UNet()
    model.load_state_dict(model_state_dict)

    # 加载数据集
    image_dir = '2_Train'
    json_files = get_json_files_in_folder('2_Train')
    dataset = OracleBoneDataset(image_dir, json_files)
    
    print("数据加载完成")
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False) # 创建数据加载器  shuffle 为 False，因为在评估阶段不需要打乱数据顺序

    model.eval() # 将模型设置为评估模式

    # 定义评估指标
    # 这里假设你有一些评估指标函数，比如计算准确率、IoU 等
    # 准备变量来保存预测结果和真实标签，用于后续的评估
    all_predictions = []
    all_labels = []
    # 初始化
    ious, dices, pixel_accs = [], [], []

    with torch.no_grad():
        for images, masks, img_names,_ in data_loader:

            image_path = os.path.join('2_Train', img_names[0]) 
            image_ori = cv2.imread(image_path)

            
            height_original, width_original, _ = image_ori.shape # 注意：cv2中图像尺寸的顺序是(高度, 宽度)
            
            width_scale = width_original / 128
            height_scale = height_original / 128
            
            outputs = model(images) # 运行模型
            print("模型运行完成")
            preds = (outputs >= outputs_threshold).float()  # 进行二值化时的阈值，现在的preds转变成了0/1
            
            preds_int = preds.byte() # 确保preds和masks都是整数类型，以便进行位运算
            masks_int = masks.byte() # .byte()用于将浮点Tensor转换为byte类型Tensor

            image_data = preds.squeeze().cpu().numpy()  # 转换为NumPy数组
            
            if 1 == 1: # 选择是否展示图片
                plt.imshow(image_ori, cmap='gray')
                plt.axis('off')
                plt.show()
            
            preds_np = preds.squeeze().cpu().numpy() # 转换预测结果为NumPy数组，并确保它是二值的（0或255）
            preds_np = (preds_np * 255).astype(np.uint8)  # 转换为8-bit数组，值为0或255
        
            # 寻找轮廓 (注意：OpenCV 4不返回第二个元素'hierarchy')
            contours, _ = cv2.findContours(preds_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
            # 遍历轮廓并绘制
            for contour in contours:
                # 最小外接矩形
                x, y, w, h = cv2.boundingRect(contour)
                if (w * h) > 25:  # 只有当面积大于指定像素时才绘制
                    x1_original = int(x * width_scale)
                    y1_original = int(y * height_scale)
                    x2_original = int((x+w) * width_scale)
                    y2_original = int((y+h) * height_scale)
                    cv2.rectangle(image_ori, (x1_original, y1_original), (x2_original, y2_original), (0,0,255), thickness=1)  # 红色线
           

            # 显示灰度图像
            if 1 == 1:        
                plt.imshow(preds_np, cmap='gray')
                plt.axis('off')
                plt.show()
            
                image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
                plt.imshow(image_ori)
                plt.axis('off')
                plt.show()


            # 计算一些评估指标
            ious.append(calculate_iou(preds_int, masks_int))
            dices.append(calculate_dice(preds_int, masks_int))
            pixel_accs.append(calculate_pixel_accuracy(preds_int, masks_int)) # 假设pixel_accs是一个列表，列表中的每个元素都是一个包含单个像素准确度值的张量

    # ------------------------计算整体评估指标------------------------
    if 1 == 1:
        mean_iou = torch.mean(torch.tensor(ious))
        mean_dice = torch.mean(torch.tensor(dices))
        mean_pixel_acc = torch.mean(torch.stack(pixel_accs)) # 可以使用torch.stack来沿着一个新维度堆叠所有这些张量，然后计算它们的均值
    
        print("------当前参数------")
        print(f"图片数量: {picture_num}")
        print(f"UNet迭代次数: {num_epochs}")
        print(f"输出分割二值化时的阈值: {outputs_threshold:.2f}")
        print("------模型评估------")
        print(f'平均IoU: {mean_iou:.4f}')
        print(f'平均Dice系数: {mean_dice:.4f}')
        print(f'平均像素准确率: {mean_pixel_acc:.4f}')

        output_line = (f"图片数量: {picture_num}, "
                       f"UNet迭代次数: {num_epochs}, "
                       f"输出分割二值化时的阈值: {outputs_threshold:.2f}, "
                       f"平均IoU: {mean_iou:.4f}, "
                       f"平均Dice系数: {mean_dice:.4f}, "
                       f"平均像素准确率: {mean_pixel_acc:.4f}\n")

        with open('2_results/results.txt', 'a') as file:
            file.write(output_line)
        
        outputs_threshold -= 0.05