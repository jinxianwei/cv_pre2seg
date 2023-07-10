from transforms import LoadImageFromFile, Classic_ThresholdAlgorithm, Classic_Edges, Classic_BaseAlgorithm, Classic_Extrema


def main():
    path_set = [
        'E:\\cv_pre2seg\\source\\1_8bit.jpg',
            'E:\\cv_pre2seg\\source\\2.jpg',
            'E:\\cv_pre2seg\\source\\3.jpg',
            'E:\\cv_pre2seg\\source\\4.png',
            'E:\\cv_pre2seg\\source\\5.tif',
        'E:\\cv_pre2seg\\source\\moeda.png',
        'E:\\cv_pre2seg\\source\\ocr.jpg',
        'E:\\cv_pre2seg\\source\\facial2.jpg'
            ]
    # path_set = ['/mnt/e/cv_pre2seg/source/1_8bit.jpg',
    #             '/mnt/e/cv_pre2seg/source/2.jpg',
    #             '/mnt/e/cv_pre2seg/source/3.jpg',
    #             '/mnt/e/cv_pre2seg/source/4.png',
    #             '/mnt/e/cv_pre2seg/source/5.tif']
    
    for i in range(len(path_set)):
        img_path = path_set[i]
        result = dict()

        LoadImage = LoadImageFromFile(img_path)
        result = LoadImage.transform(result)
        # print(result)
        

        # threshold_al = Classic_ThresholdAlgorithm(result)
        # threshold_al.emThreshold()
        # threshold_al.adaptiveThreshold()
        # threshold_al.rangeThreshold(91, 208)
        # threshold_al.basicThreshold(125)

        classic_edge = Classic_Edges(result)
        # classic_edge.Advanced_FindText()
        classic_edge.Advanced_FindFacialFeatures()
        # classic_edge.Find_Edges('Canny')
        # classic_edge.Find_Edges('Sobel')
        # classic_edge.Find_Edges('Laplacian')
        # classic_edge.Find_Lines()
        # classic_edge.Watershed()
        # classic_edge.Find_Circles()

        # classic_invert = Classic_BaseAlgorithm(result)
        # classic_invert.base_Invert()
        # classic_invert.base_Blank()

        # classic_Exream = Classic_Extrema(result)
        # classic_Exream.Extrema_globalMaximum()
        # classic_Exream.Extrema_globalMinimum()
        # classic_Exream.Extrema_LocalMaxima(122)

    return 0

if __name__ == '__main__':
    main()