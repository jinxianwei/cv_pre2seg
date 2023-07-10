from PIL import Image

# 打开TIFF图像
image = Image.open('E:\\cv_pre2seg\\source\\5.tif')

# 获取TIFF图像的所有标签
tags = image.tag_v2

# 获取Orientation标签的值
if 'Orientation' in tags:
    orientation = tags['Orientation']
    print(f"图像的Orientation为 {orientation}")
else:
    print("图像中没有Orientation标签")

# 关闭图像
image.close()
