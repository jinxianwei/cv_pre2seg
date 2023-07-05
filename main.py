from transforms import LoadImageFromFile


def main():
    path_set = ['E:\cv_pre2seg\source\ES32_HR_1000x1000x1001_8bit.dat.jpg',
            'E:\cv_pre2seg\source\SAVII2_m_0162.tif.thumb.jpg',
            'E:\cv_pre2seg\source\SAVII2_mid_slice500_scaleBar.png',
            'E:\cv_pre2seg\source\SAVII2_m_0162.tif']
    # jpg -> BGR  png -> BGRA(默认将透明度通道去掉了)  shape[0]= h shape[1]=w shape[2]=c

    img_path = path_set[3]
    result = dict()

    LoadImage = LoadImageFromFile(img_path)
    result = LoadImage.transform(result)
    print(result)

    return 0

if __name__ == '__main__':
    main()