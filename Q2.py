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
class UNet(nn.Module): # ��ģ��
    def __init__(self):
        super(UNet, self).__init__()
        # ������
        self.enc_conv1 = self.encoder_block(1, 8)
        self.enc_conv2 = self.encoder_block(8, 16)
        self.enc_conv3 = self.encoder_block(16, 32)
        self.enc_conv4 = self.encoder_block(32, 128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # ������
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv4 = self.decoder_block(192, 64)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv3 = self.decoder_block(64, 32)

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv2 = self.decoder_block(32, 16)
        
        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec_conv1 = self.decoder_block(16, 8)
        # ���ղ�
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
        # ����
        enc1 = self.enc_conv1(x)
        pool1 = self.pool(enc1)
        
        enc2 = self.enc_conv2(pool1)
        pool2 = self.pool(enc2)
        
        enc3 = self.enc_conv3(pool2)
        pool3 = self.pool(enc3)
        
        enc4 = self.enc_conv4(pool3)
        pool4 = self.pool(enc4)
        
        # ����
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
        # ���ղ�
        return torch.sigmoid(self.final_conv(dec1))
# ������Ѱ������ڽӾ���
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
''' # �ɵ�ģ��
def get_json_files_in_folder(folder_path): # ��ȡ����ļ����µ�����json�ļ������б���ʽ���
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
    # ���㽻��
    intersection = (preds & labels).float().sum((1, 2))
    # ���㲢��
    union = (preds | labels).float().sum((1, 2))
    iou = intersection / (union + 1e-6)
    return iou.mean()  # ��������ƽ��IoU

def calculate_dice(preds, labels):
    # ����Diceϵ��
    intersection = (preds & labels).float().sum((1, 2))
    dice = (2 * intersection) / (preds.float().sum((1, 2)) + labels.float().sum((1, 2)) + 1e-6)
    return dice.mean()  # ��������ƽ��Diceϵ��

def calculate_pixel_accuracy(preds, labels):
    # ��������׼ȷ��
    correct = (preds == labels).float().sum((1, 2))
    total = labels.numel() / labels.size(0)  # ÿ��ͼ�����������
    return correct / total  # ��������ƽ��Pixel Accuracy

class AttentionGate(nn.Module): # ע��������
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
        # ����
        enc1 = self.enc_conv1(x)
        pool1 = self.pool(enc1)
    
        enc2 = self.enc_conv2(pool1)
        pool2 = self.pool(enc2)
    
        enc3 = self.enc_conv3(pool2)
        pool3 = self.pool(enc3)
    
        enc4 = self.enc_conv4(pool3)
        pool4 = self.pool(enc4)
    
        # ����
        up4 = self.upconv4(pool4)
        # ע�������Ƽ���ǰ��merge��Ϊʹ��ע�����ź���ں�
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
        # ���ղ�
        return torch.sigmoid(self.final_conv(dec1))

class UNet(nn.Module): # ���ĵĻ���ѧϰ����
    def __init__(self):
        super(UNet, self).__init__()
        # ������
        self.enc_conv1 = self.encoder_block(1, 8)
        self.enc_conv2 = self.encoder_block(8, 16)
        self.enc_conv3 = self.encoder_block(16, 32)
        self.enc_conv4 = self.encoder_block(32, 128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # ������
        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv4 = self.decoder_block(192, 64)

        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_conv3 = self.decoder_block(64, 32)

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec_conv2 = self.decoder_block(32, 16)
        
        self.upconv1 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)
        self.dec_conv1 = self.decoder_block(16, 8)
        # ���ղ�
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
        # ����
        enc1 = self.enc_conv1(x)
        pool1 = self.pool(enc1)
        
        enc2 = self.enc_conv2(pool1)
        pool2 = self.pool(enc2)
        
        enc3 = self.enc_conv3(pool2)
        pool3 = self.pool(enc3)
        
        enc4 = self.enc_conv4(pool3)
        pool4 = self.pool(enc4)
        
        # ����
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
        # ���ղ�
        return torch.sigmoid(self.final_conv(dec1))

class OracleBoneDataset(Dataset): # �������ݼ�
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

        # ��ԭʼmask����Ϊ128x128���ش�С
        resized_mask = np.array(Image.fromarray(mask.squeeze(0).numpy()).resize((128, 128)))

        # ��������С���mask�����µ�128x128������
        new_mask[:resized_mask.shape[0], :resized_mask.shape[1]] = resized_mask

        # ���µ�maskת��ΪPyTorch����
        new_mask = torch.tensor(new_mask).unsqueeze(0)
        threshold = 0.5
        new_mask[new_mask < threshold] = 0
        new_mask[new_mask >= threshold] = 1

        # ���ӻ���ֵ�����mask
        # plt.imshow(new_mask.squeeze(0), cmap='gray')
        # plt.title('Binary Mask')
        # plt.show()

        return image_x, new_mask, img_name, img_ann

picture_num = 6148 # ����ģ��ʱ���ܹ������ͼƬ����
num_epochs = 25 # ����ģ��ʱ����������
outputs_threshold = 0.5 # ����ģ��ʱ�������ֵ�����ֵ��ʼ����0.05�ݼ�
model_path = 'unet_model_6148_25_att.pth' # ѡ��ģ�ͱ���/ʹ�õ�·��

cs = 2 # 1, ѵ��ģ��
       # 2, ����ģ��



if cs == 1: # ѵ��ģ��
    
    image_dir = '2_Train' # ʵ�������ݼ� ��Ҫ�滻���Լ���·��
    # json_files = ['2_Train_bf/b02519Z.json', '2_Train_bf/b02523F.json']
    json_files = get_json_files_in_folder(image_dir)
    dataset = OracleBoneDataset(image_dir, json_files)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False) # ���ݼ�����
    print("���ݼ������")
    model = UNet() # ʵ����ģ��
    print("ģ��ʵ�������")
    criterion = nn.BCELoss() # ��ʧ����
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # �Ż���
    print("��ʧ�������Ż����������")
    # ѵ��ģ��
    for epoch in range(num_epochs):
        print(f"��ʼ��{epoch + 1}/{num_epochs}��ѵ��ģ��")
        model.train()
        for images, masks, _, _  in data_loader:

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), model_path) # ����ģ��Ȩ��
    print("Model training complete and saved.")
    
while outputs_threshold > 0.001 and cs == 2: # ����ģ��
    model_state_dict = torch.load(model_path)

    # ʵ����ģ��
    model = UNet()
    model.load_state_dict(model_state_dict)

    # �������ݼ�
    image_dir = '2_Train'
    json_files = get_json_files_in_folder('2_Train')
    dataset = OracleBoneDataset(image_dir, json_files)
    
    print("���ݼ������")
    
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False) # �������ݼ�����  shuffle Ϊ False����Ϊ�������׶β���Ҫ��������˳��

    model.eval() # ��ģ������Ϊ����ģʽ

    # ��������ָ��
    # �����������һЩ����ָ�꺯�����������׼ȷ�ʡ�IoU ��
    # ׼������������Ԥ��������ʵ��ǩ�����ں���������
    all_predictions = []
    all_labels = []
    # ��ʼ��
    ious, dices, pixel_accs = [], [], []

    with torch.no_grad():
        for images, masks, img_names,_ in data_loader:

            image_path = os.path.join('2_Train', img_names[0]) 
            image_ori = cv2.imread(image_path)

            
            height_original, width_original, _ = image_ori.shape # ע�⣺cv2��ͼ��ߴ��˳����(�߶�, ���)
            
            width_scale = width_original / 128
            height_scale = height_original / 128
            
            outputs = model(images) # ����ģ��
            print("ģ���������")
            preds = (outputs >= outputs_threshold).float()  # ���ж�ֵ��ʱ����ֵ�����ڵ�predsת�����0/1
            
            preds_int = preds.byte() # ȷ��preds��masks�����������ͣ��Ա����λ����
            masks_int = masks.byte() # .byte()���ڽ�����Tensorת��Ϊbyte����Tensor

            image_data = preds.squeeze().cpu().numpy()  # ת��ΪNumPy����
            
            if 1 == 1: # ѡ���Ƿ�չʾͼƬ
                plt.imshow(image_ori, cmap='gray')
                plt.axis('off')
                plt.show()
            
            preds_np = preds.squeeze().cpu().numpy() # ת��Ԥ����ΪNumPy���飬��ȷ�����Ƕ�ֵ�ģ�0��255��
            preds_np = (preds_np * 255).astype(np.uint8)  # ת��Ϊ8-bit���飬ֵΪ0��255
        
            # Ѱ������ (ע�⣺OpenCV 4�����صڶ���Ԫ��'hierarchy')
            contours, _ = cv2.findContours(preds_np, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
            # ��������������
            for contour in contours:
                # ��С��Ӿ���
                x, y, w, h = cv2.boundingRect(contour)
                if (w * h) > 25:  # ֻ�е��������ָ������ʱ�Ż���
                    x1_original = int(x * width_scale)
                    y1_original = int(y * height_scale)
                    x2_original = int((x+w) * width_scale)
                    y2_original = int((y+h) * height_scale)
                    cv2.rectangle(image_ori, (x1_original, y1_original), (x2_original, y2_original), (0,0,255), thickness=1)  # ��ɫ��
           

            # ��ʾ�Ҷ�ͼ��
            if 1 == 1:        
                plt.imshow(preds_np, cmap='gray')
                plt.axis('off')
                plt.show()
            
                image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
                plt.imshow(image_ori)
                plt.axis('off')
                plt.show()


            # ����һЩ����ָ��
            ious.append(calculate_iou(preds_int, masks_int))
            dices.append(calculate_dice(preds_int, masks_int))
            pixel_accs.append(calculate_pixel_accuracy(preds_int, masks_int)) # ����pixel_accs��һ���б��б��е�ÿ��Ԫ�ض���һ��������������׼ȷ��ֵ������

    # ------------------------������������ָ��------------------------
    if 1 == 1:
        mean_iou = torch.mean(torch.tensor(ious))
        mean_dice = torch.mean(torch.tensor(dices))
        mean_pixel_acc = torch.mean(torch.stack(pixel_accs)) # ����ʹ��torch.stack������һ����ά�ȶѵ�������Щ������Ȼ��������ǵľ�ֵ
    
        print("------��ǰ����------")
        print(f"ͼƬ����: {picture_num}")
        print(f"UNet��������: {num_epochs}")
        print(f"����ָ��ֵ��ʱ����ֵ: {outputs_threshold:.2f}")
        print("------ģ������------")
        print(f'ƽ��IoU: {mean_iou:.4f}')
        print(f'ƽ��Diceϵ��: {mean_dice:.4f}')
        print(f'ƽ������׼ȷ��: {mean_pixel_acc:.4f}')

        output_line = (f"ͼƬ����: {picture_num}, "
                       f"UNet��������: {num_epochs}, "
                       f"����ָ��ֵ��ʱ����ֵ: {outputs_threshold:.2f}, "
                       f"ƽ��IoU: {mean_iou:.4f}, "
                       f"ƽ��Diceϵ��: {mean_dice:.4f}, "
                       f"ƽ������׼ȷ��: {mean_pixel_acc:.4f}\n")

        with open('2_results/results.txt', 'a') as file:
            file.write(output_line)
        
        outputs_threshold -= 0.05