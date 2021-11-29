**Extraction of Interest Points and Establishing Correspondences using Harris Corner Detector and SIFT Operator**

**1. Introduction:**

In this homework we used a pair of images taken from the same scene but
from different perspectives/viewpoints. We extracted the interest points
in the images using Harris corner and SIFT feature extractor from
OpenCV. Finally, automatic correspondences among the corners of a given
image pair were established using sum of squared differences (SSD) and
normalized cross correlation (NCC). For the SIFT features, we performed
brute force correspondences. Details of the methods can be seen in the
attached pdf report.


**Results:**


1.  **Harris Corner:**

<img src="media\image1.jpeg" style="width:5.82292in;height:3.27539in" /><img src="media\image2.jpeg" style="width:5.83333in;height:3.28125in" />

**Harris corners from images in pair 1 at sigma = 0.6; k = 0.04,
noise\_window = 15.**

<img src="media\image3.jpeg" style="width:5.83111in;height:3.28in" /><img src="media\image4.jpeg" style="width:5.83111in;height:3.28in" />

**Harris corners from images in pair 1 at sigma = 2.4; k = 0.04,
noise\_window = 15.**



2.  **Correspondence among Harris corners using SSD:**

<img src="media\image5.jpeg" style="width:6.07292in;height:1.70833in" />

**SSD on Harris corners from images in pair 1 at sigma = 0.6; k = 0.04,
noise\_win= 15.**

<img src="media\image6.jpeg" style="width:6.07885in;height:1.71in" />

**SSD on Harris corners from images in pair 1 at sigma = 2.4; k = 0.04,
noise\_win= 15.**

<img src="media\image7.jpeg" style="width:6.07885in;height:1.71in" />

**NCC on Harris corners from images in pair 1 at sigma = 0.6;k = 0.04,
noise\_win= 15.**

3.  **Correspondence among Harris corners using NCC:**

<img src="media\image7.jpeg" style="width:6.07885in;height:1.71in" />

> **NCC on Harris corners from images in pair 1 at sigma = 0.6;k = 0.04,
> noise\_win= 15.**

<img src="media\image8.jpeg" style="width:6.07885in;height:1.71in" />

**NCC on Harris corners from images in pair 1 at sigma =2.4;k =
0.04,noise\_win= 15.**

4.  **SIFT features and their descriptors:**

<img src="media\image9.jpeg" style="width:6.5in;height:3.65625in" />

<img src="media\image10.jpeg" style="width:6.5in;height:3.65625in" />

**SIFT features from images in pair 1 with contrast thresh =0.1 & max
features =5000.**

5.  **Correspondence among SIFT features using brute force matching:**

<img src="media\image11.jpeg" style="width:6.5in;height:1.82847in" />

**Fig 2: Corresponding SIFT features from images in pair 1 with contrast
thresh =0.1 & max features =5000 and brute force matching.**
