# -*- coding: gbk -*-
import os
import cv2  # ����cv2ģ��

output_folder='1_After_test'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

folder = '1_Pre_test'

img_names = [img_name for img_name in os.listdir(folder)]

for img_name in img_names:
    img_path = os.path.join(folder,img_name)
    # ��ȡͼ��
    image = cv2.imread(img_path)  # ʹ�õ�ǰѭ����������ͼ���ļ�����ȡͼ��
    
    # ��ͼ��ת��Ϊ�Ҷ�ͼ
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # ʹ��cv2.cvtColor������ͼ���BGRɫ�ʿռ�ת��Ϊ�Ҷ�ͼ
    # ѡ��һ�����ʵ���ֵ��������127Ϊ��
    threshold_value = 105  # ������ֵΪ127
    # ��ͼ����ж�ֵ��
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)  # ʹ��cv2.threshold�������ж�ֵ������
    
    medianFiltered_image = cv2.medianBlur(binary_image, 5) # ʹ����ֵ�˲�ȥ����״����

    averageFiltered_image = cv2.blur(binary_image, (5,5))  # ʹ�þ�ֵ�˲�ȥ����״����

    # �ԻҶ�ͼ����зǾֲ���ֵȥ��
    denoised_image1 = cv2.fastNlMeansDenoising(gray_image, None, h=10, templateWindowSize=21, searchWindowSize=21)
    """
    ʹ�÷Ǿֲ���ֵ�㷨��ͼ�����ȥ��
    :param image: �����ͼ��
    :param h: ����ƽ���̶ȵĲ���
    :param search_window: �������ڵĴ�С
    """
    _, denoised_image = cv2.threshold(denoised_image1, threshold_value, 255, cv2.THRESH_BINARY)

    # ��ʾԭʼͼ��Ͷ�ֵ�����ͼ��
    cv2.imshow('Original Image', image)  # ʹ��cv2.imshow������ʾԭʼͼ��
    cv2.imshow('Binary Image', binary_image)  # ʹ��cv2.imshow������ʾ ��ֵ����� ͼ��
    cv2.imshow('medianFiltered_image', medianFiltered_image)  # ʹ��cv2.imshow������ʾ ��ֵ�˲�ȥ����״������� ��ֵ����� ͼ��
    cv2.imshow('denoised_image', denoised_image)  # ʹ��cv2.imshow������ʾ ��ֵ�˲�ȥ����״������� ��ֵ����� ͼ��

    # ����Ҫ������ļ�����·��
    output_file1 = os.path.join(output_folder, 'Binary_image_' + img_name)
    output_file2 = os.path.join(output_folder, 'medianFiltered_image_' + img_name)
    output_file3 = os.path.join(output_folder, 'denoised_image_' + img_name)

    # ʹ��imwrite��������ͼ��
    cv2.imwrite(output_file1, binary_image)
    cv2.imwrite(output_file2, medianFiltered_image)
    cv2.imwrite(output_file3, denoised_image)

    cv2.waitKey(0)  # �ȴ��û���������
    cv2.destroyAllWindows()  # �ر�����OpenCV�����Ĵ���