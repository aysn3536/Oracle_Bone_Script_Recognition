# -*- coding: gbk -*-
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from openpyxl import load_workbook
# image_ori = cv2.imread(os.path.join('4_Recognize','test_set','w01790.jpg'))
# cv2.imshow('Image', image_ori)

# cv2.waitKey(0)
# # 关闭窗口
# cv2.destroyAllWindows()

picture_num = 6148 # 生成模型时，总共处理的图片数量
num_epochs = 25 # 生成模型时，迭代次数
outputs_threshold = 0.36 # 运行模型时的输出阈值从这个值开始，以0.05递减

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

class UnlabeledImageDataset(Dataset): # 没有标签的数据集
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.transform = transforms.ToTensor()
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert('L') # 以灰度图像读取
        
        image = image.resize((128, 128)) # 压缩尺寸
        
        image = self.transform(image) # 转化为张量
        
        return image, img_name
image_dir = os.path.join('4_Recognize','test_set')  # 更新图片目录路径
    
def find_value_row(filename, sheetname, value):
    # 打开Excel文件
    wb = load_workbook(filename)
    
    # 选择工作表
    ws = wb[sheetname]
    i = 1
    # 遍历第一列，查找值所在的行号
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=1, values_only=True):
        if row[0] == value:
            print(i)
            return i  # 直接返回行号
        else:
            i += 1
    # 如果值未找到，返回None
    return None
unlabeled_dataset = UnlabeledImageDataset(image_dir) # 加载数据

# 创建数据加载器
data_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False)

model_path = 'unet_model_6148_25_att.pth' # 加载模型
model_state_dict = torch.load(model_path)
model = UNet()
model.load_state_dict(model_state_dict)
    
model.eval() # 设置为评估模式

with torch.no_grad():
    for images, img_name in data_loader:
                        
        image_name = img_name[0]
            
        img_path = os.path.join(image_dir, image_name) # 获取图片路径
        print(img_path)
        image_ori = cv2.imread(img_path) # 读取图片（原始图片）
            
        height_original, width_original, _ = image_ori.shape # 注意：cv2中图像尺寸的顺序是(高度, 宽度)

        width_scale = width_original / 128 # 获取原始图片的缩放比例
            
        height_scale = height_original / 128 # 获取原始图片的缩放比例
            
        outputs = model(images)  # 使用模型进行预测
            
        preds = (outputs >= outputs_threshold).float()  # 应用阈值
            
        image_data = preds.squeeze().cpu().numpy()  # 转换为NumPy数组，这个image_data在之后可以展示出来
            
        preds_np = preds.squeeze().cpu().numpy() # 转换为NumPy数组
            
        preds_np = (preds_np * 255).astype(np.uint8)  # 转换为8-bit数组，值为0或255
            
        contours, _ = cv2.findContours(preds_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # 获取最小外接矩形
            
        character_rect_list = []
            
        for contour in contours: # 遍历轮廓并绘制矩形
                
            x, y, w, h = cv2.boundingRect(contour) # 最小外接矩形
            if (w * h) > 22:  # 只有当面积大于指定大小时才绘制
                x1_original = int(x * width_scale) - 1 # 左上角坐标，并映射到原始的图像坐标中
                y1_original = int(y * height_scale) - 1 
                x2_original = int((x+w) * width_scale) + 1 # 右下角坐标
                y2_original = int((y+h) * height_scale) + 1
                    
                character_rect_list.append([x1_original,y1_original,x2_original,y2_original, 1.0])
                    
                
                if 1 == 2: # 是否保存裁切后的图片

                    cropped_image = image_ori[y1_original:y2_original, x1_original:x2_original]
                    cropped_image_name = f"{image_name[:-4]}_{x1_original}_{y1_original}_{x2_original}_{y2_original}.jpg"

                    # 保存提取的图像
                    cropped_image_save_path = os.path.join('saved_crops', cropped_image_name)

                    cv2.imwrite(cropped_image_save_path, cropped_image)

                cv2.rectangle(image_ori, (x1_original, y1_original), (x2_original, y2_original), (0,0,255), thickness=1)  # 红色线

        print(str(character_rect_list))

        if 1 == 2: # 判断是否将裁切结果写入excel表格

            wb = load_workbook(os.path.join("3_Test", "Test_results.xlsx"))
            ws = wb['Sheet1']
            # for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=1, values_only=True):
            #     if row[0] == image_name:
            #         print(row[0])
            #         ws.cell(row[0].row, column=2, value=str(character_rect_list))
            row_number = find_value_row(os.path.join("3_Test", "Test_results.xlsx"), 'Sheet1', image_name)
            ws.cell(row = row_number, column = 2, value = str(character_rect_list)[1: -1]) 
            wb.save(os.path.join("3_Test", "Test_results.xlsx"))
        if 1 == 2: # 选择是否保存图片（绘制红色矩形）
                
            save_path = os.path.join('saved_crops','ori_pics',img_name[0]) # 拼贴图片保存路径
            
            cv2.imwrite(save_path, image_ori) # 保存输出结果（在原图片上绘制矩形的结果）
            
        if 1 == 1: # 选择是否显示图像

            image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)

            plt.imshow(preds_np, cmap='gray') # 显示2值化的预测范围
            plt.axis('off')
            plt.show()
            
            plt.imshow(image_ori) # 在原图片上绘制红色矩形的图片
            plt.axis('off')
            plt.show()

            plt.imshow(image_data, cmap='gray') # 原始图片
            plt.axis('off')
            plt.show()
                
            