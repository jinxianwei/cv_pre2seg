# cv_pre2seg

## Add image properites using opencv-python

- [x] size
- [x] File size
- [x] Width
- [x] Height
- [x] BitDepth
- [x] MaxSamplevalue
- [x] MinSamplevalue
- [o] NewSubFileType
- [o] BitsPerSample
- [o] SamplePrePixel
- [o] RowsPerStrip
- [ ] XResolution
- [ ] YRsolution
- [o] TileWidth
- [o] TileLength
- [ ] Orientation
- [o] FillOrder
- [ ] GrayResponseunit
- [ ] Thresholding
- [o] Offset

## Some segmentation using Classic algorithm

- [x] Base Invert
- [x]      Blank  貌似需要为其他算法提供图层功能
- [x] Threshold Basic Threshold (%的方式未复现)
- [x]           Range Threshold
- [x]           Adaptive Threshold
- [ ]           E-M Threshold
- [x] Edges Watershed  待优化
- [o]       Find Edges 除了canny的laplacian和sobel边缘都有非边缘图像，是否需要进行二值化
- [x]       Find Circles
- [x]       Find Lines
- [x]       Advanced FindText  尝试以deepLearning进行inference
- [x]                Find Facial Features
- [ ] SNAP Auto Segmentation
- [x] Extrema Find global Maximum
- [x]         Find global Minimum
- [ ]         Find Local Maxima  貌似有模糊的操作
- [ ]         Find Local Minima
- [ ] Other Machine Learning model
