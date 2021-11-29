**Frame Camera Calibration using Zhnag’s algorithm**

This homework performs runs the camera calibration routines on the set
of images containing the checkboard pattern. Two datasets were used for
this homework. One dataset was provided with this homework and other was
created as the homework instructions. The implemented algorithm has
following steps:

1.  First detect the edges of all the boxes in a checkerboard pattern
    using the Canny edge detector. We then use the Hough transformation
    to extract the horizontal and vertical lines joining the edges. The
    intersection of the lines yields corners in an image.

2.  Then we calculated the homography matrices between image coordinates
    and the world coordinates of the boxes on checkerboard pattern.
    These homographies were finally used with Zhang’s algorithm \[1)\]
    to estimate the intrinsic and extrinsic parameters of the camera and
    images, respectively.

3.  The intrinsic and extrinsic parameters were then refined using the
    Levenburg-Marquardt (LM) algorithm to compensate for the
    non-linearities.

**Results:**

<img src="media\image1.jpeg" style="width:5.70235in;height:3.5in" />

**(a)**

<img src="media\image2.jpeg" style="width:5.72524in;height:3.5in" />

**(b)**

<img src="media\image3.jpg" style="width:5.68194in;height:3.53125in" alt="Letter Description automatically generated" />

**(c)**

**Edges, lines and corners of image 3, (a) edges, (b) Hough lines and
(c) corners**

<img src="media\image4.jpeg" style="width:5.97545in;height:3.7in" />

**(a)**

<img src="media\image5.jpeg" style="width:5.89241in;height:3.56in" />

**(b)**

<img src="media\image6.jpg" style="width:6in;height:4in" alt="Letter Description automatically generated" />

**(c)**

**Edges, lines and corners of image 21, (a) edges (b) Hough lines and
(c) corners**

<img src="media\image7.jpeg" style="width:5.83255in;height:3.53264in" />

**(a)**

<img src="media\image8.jpeg" style="width:5.72847in;height:3.56422in" />

**(b)**

**Reprojection of the corners in image 15 onto the fixed image (image
11). The green dots represent the actual corners, while the red
represents the reprojections (a) using linear least squares (mean error
= 5.2790 , variance = 24.8069) and (b) using LM refinement (mean error =
0.6704, variance = 0.2525). Labelling scheme is same as shown in the
earlier images.**

<img src="media\image9.jpeg" style="width:5.84589in;height:3.56in" />

**(a)**

<img src="media\image10.jpeg" style="width:5.86132in;height:3.56in" />

**(b)**

**Fig 5: Reprojection of the corners in image 21 onto the fixed image
(image 11). The green dots represent the actual corners, while the red
represents the reprojections (a) using linear least squares (mean error
= 7.6409 , variance = 18.7880) and (b) using LM refinement (mean error =
0.9854, variance = 0.8225). Labelling scheme is same as shown in the
earlier images.**

**Intrinsic parameters for the given dataset:**

<img src="media\image11.png" style="width:6.5in;height:3.97292in" alt="Text Description automatically generated" />

**Extrinsic parameters for the given dataset:**

<img src="media\image12.png" style="width:6.5in;height:4.33681in" alt="A screenshot of a computer Description automatically generated with low confidence" />

**Extrinsic parameters with LM and distortion incorporated:**

<img src="media\image13.png" style="width:5.6875in;height:1.29167in" alt="Table Description automatically generated" />

**Camera pose for fixed imaged with measured ground-truth :**

The \[*R*<sub>11</sub>\|*t⃗*<sub>11</sub>\]for the fixed image 11 with
LLS and LM are given below. It can be seen that R is approximately
identity matrix with t representing the translation.

<img src="media\image14.png" style="width:5.6875in;height:1.77083in" alt="Table Description automatically generated" />

**References:**

1.  Zhengyou Zhang. A flexible new technique for camera calibration.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    22:1330-1334, December 2000. MSR-TR-98-71, Updated March 25, 1999.
