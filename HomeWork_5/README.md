**Image stitching using SIFT features, RANSAC algorithm with linear and
non-linear Homography estimation/ refinement**

This homework generates panorama using five overlapping images. The
brief overview of the method used is as follows:

1.  In the first step, SIFT operator from the OpenCV was used to extract
    the key features of an image. The SIFT features were then used
    together with normalized cross-correlation (NCC) to establish the
    correspondences between an image pair.

2.  In order to remove the false correspondences, we used the random
    sample consensus (RANSAC) algorithm to find only the inliers. These
    inliers were then used to find the homographies using linear least
    squares approach.

3.  Finally, we used Levenberg-Marquardt (LM) algorithm to further
    refine the homogrpahies in non-linear fashion. The LM method used
    the homographies estimated in earlier step as initial guess.

**Results:**

1.  **Input images:**

<img src="media\image1.jpeg" style="width:3.33333in;height:2.5in" />

**(a)**

<img src="media\image2.jpeg" style="width:3.33333in;height:2.5in" />

**(b)**

<img src="media\image3.jpeg" style="width:3.33333in;height:2.5in" />

**(c)**

<img src="media\image4.jpeg" style="width:3.33333in;height:2.5in" />

**(d)**

<img src="media\image5.jpeg" style="width:3.33333in;height:2.5in" />

**(e)**

**Set of 5 input images used for this homework. Images were captured
with the help of RaspberryPi camera for the corn leaf.**

2.  **SIFT features and resulting NCC correspondences:**

<img src="media\image6.jpeg" style="width:6in;height:4in" />

**Correspondences detected by SIFT with NCC between image pair a and
b.**

<img src="media\image7.jpeg" style="width:6in;height:4in" />

**Correspondences detected by SIFT with NCC between image pair b and
c.**

<img src="media\image8.jpeg" style="width:6in;height:4in" />

**Correspondences detected by SIFT with NCC between image pair c and
d.**

<img src="media\image9.jpeg" style="width:6in;height:4in" />

**Correspondences detected by SIFT with NCC containing image pair d and
e.**

3.  **Correspondences rejected by RANSAC:**

<img src="media\image10.jpeg" style="width:6in;height:4in" />

**Correspondences rejected by RANSAC algorithm (Red) between image pair
a and b.**

<img src="media\image11.jpeg" style="width:6in;height:4in" />

**Correspondences rejected by RANSAC algorithm (Red) between image pair
b and c.**

<img src="media\image12.jpeg" style="width:6in;height:4in" />

**Correspondences rejected by RANSAC algorithm (Red) between image pair
c and d.**

<img src="media\image13.jpeg" style="width:6in;height:4in" />

**Correspondences rejected by RANSAC algorithm (Red) between image pair
d and e.**

4.  **Stitched images after homography estimation using Least Squares:**

<img src="media\image14.jpeg" style="width:3.5805in;height:4in" />

**Panorama generated by using the homographies estimated with the help
of RANSAC algorithm.**

5.  **Stitched images after homography refinement using
    Levenburg-Marquardt algorithm:**

<img src="media\image15.jpeg" style="width:3.5805in;height:4in" />

**Panorama generated by using the homographies refined using
Levenburg-Marquardt algorithm.**