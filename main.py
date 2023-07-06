from transforms import LoadImageFromFile, Classic_ThresholdAlgorithm


def main():
    path_set = ['E:\\cv_pre2seg\\source\\1_8bit.jpg',
            'E:\\cv_pre2seg\\source\\2.jpg',
            'E:\\cv_pre2seg\\source\\3.jpg',
            'E:\\cv_pre2seg\\source\\4.png',
            'E:\\cv_pre2seg\\source\\5.tif'
            ]
    
    for i in range(len(path_set)):
        img_path = path_set[i]
        result = dict()

        LoadImage = LoadImageFromFile(img_path)
        result = LoadImage.transform(result)
        print(result)

        # threshold_al = Classic_ThresholdAlgorithm(result)
        # # threshold_al.adaptiveThreshold()
        # threshold_al.rangeThreshold(91, 208)

    return 0

if __name__ == '__main__':
    main()