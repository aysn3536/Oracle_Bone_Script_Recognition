# -*- coding: gbk -*-
import os
import cv2  # 导入cv2模块

output_folder='1_After_test'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

folder = '1_Pre_test'

img_names = [img_name for img_name in os.listdir(folder)]

for img_name in img_names:
    img_path = os.path.join(folder,img_name)
    # 读取图像
    image = cv2.imread(img_path)  # 使用当前循环迭代到的图像文件名读取图像
    
    # 将图像转换为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 使用cv2.cvtColor函数将图像从BGR色彩空间转换为灰度图
    # 选择一个合适的阈值，这里以127为例
    threshold_value = 105  # 设置阈值为127
    # 将图像进行二值化
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)  # 使用cv2.threshold函数进行二值化处理
    
    medianFiltered_image = cv2.medianBlur(binary_image, 5) # 使用中值滤波去除点状噪声

    averageFiltered_image = cv2.blur(binary_image, (5,5))  # 使用均值滤波去除点状噪声

    # 对灰度图像进行非局部均值去噪
    denoised_image1 = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=21, searchWindowSize=21)
    """
    使用非局部均值算法对图像进行去噪
    :param image: 输入的图像
    :param h: 控制平滑程度的参数
    :param search_window: 搜索窗口的大小
    """
    _, denoised_image = cv2.threshold(denoised_image1, threshold_value, 255, cv2.THRESH_BINARY)

    # 显示原始图像和二值化后的图像
    cv2.imshow('Original Image', image)  # 使用cv2.imshow函数显示原始图像
    cv2.imshow('Binary Image', binary_image)  # 使用cv2.imshow函数显示 二值化后的 图像
    cv2.imshow('medianFiltered_image', medianFiltered_image)  # 使用cv2.imshow函数显示 中值滤波去除点状噪声后的 二值化后的 图像
    cv2.imshow('denoised_image', denoised_image)  # 使用cv2.imshow函数显示 均值滤波去除点状噪声后的 二值化后的 图像

    # 设置要保存的文件名和路径
    output_file1 = os.path.join(output_folder, 'Binary_image_' + img_name)
    output_file2 = os.path.join(output_folder, 'medianFiltered_image_' + img_name)
    output_file3 = os.path.join(output_folder, 'denoised_image_' + img_name)

    # 使用imwrite函数保存图像
    cv2.imwrite(output_file1, binary_image)
    cv2.imwrite(output_file2, medianFiltered_image)
    cv2.imwrite(output_file3, denoised_image)

    cv2.waitKey(0)  # 等待用户按键操作
    cv2.destroyAllWindows()  # 关闭所有OpenCV创建的窗口