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

1.  Linear least-squares (LLS):

$$K = \\begin{bmatrix}
723.4997 & 1.4841 & 235.2641 \\\\
0 & 722.1189 & 319.4149 \\\\
0 & 0 & 1 \\\\
\\end{bmatrix}$$

2.  LM algorithm refined without radial distortion:

$$K = \\begin{bmatrix}
874.0656 & 1.5920 & 320.1941 \\\\
0 & 874.1790 & 233.8322 \\\\
0 & 0 & 1 \\\\
\\end{bmatrix}$$

3.  LM algorithm refined with radial distortion:

k1 = -0.17704

k2 = 0.59277

$$K = \\begin{bmatrix}
878.0351 & 1.6382 & 318.6801 \\\\
0 & 878.6688 & 232.2903 \\\\
0 & 0 & 1 \\\\
\\end{bmatrix}$$

**Extrinsic parameters for the given dataset:**

1.  Image 9 using LLS and LM:

$$\\left\\lbrack R\_{9}\|{\\overrightarrow{t}}\_{9} \\right\\rbrack\_{\\text{LLS}} = \\begin{bmatrix}
0.8840 & - 0.1378 & 0.4467 & 19.5082 \\\\
 - 0.1261 & 0.8498 & 0.5118 & - 170.1235 \\\\
 - 0.4502 & - 0.5088 & 0.7388 & 633.9512 \\\\
\\end{bmatrix}$$

$$\\left\\lbrack R\_{9}\|{\\overrightarrow{t}}\_{9} \\right\\rbrack\_{\\text{LM}} = \\begin{bmatrix}
0.8886 & - 0.0882 & 0.4844 & - 52.3874 \\\\
 - 0.1761 & 0.8409 & 0.5548 & - 88.8184 \\\\
 - 0.4598 & - 0.5754 & 0.7314 & 709.3040 \\\\
\\end{bmatrix}$$

2.  Image 12 using LLS and LM:

$$\\left\\lbrack R\_{12}\|{\\overrightarrow{t}}\_{12} \\right\\rbrack\_{\\text{LLS}} = \\begin{bmatrix}
0.9036 & 0.2036 & - 0.3769 & - 4.6823 \\\\
 - 0.0307 & 0.9083 & 0.4171 & - 170.1357 \\\\
 - 0.4273 & - 0.3653 & 0.8270 & 511.4145 \\\\
\\end{bmatrix}$$

$$\\left\\lbrack R\_{12}\|{\\overrightarrow{t}}\_{12} \\right\\rbrack\_{\\text{LM}} = \\begin{bmatrix}
0.8869 & 0.2216 & - 0.4653 & - 68.6000 \\\\
 - 0.0140 & 0.8931 & 0.5014 & - 115.1866 \\\\
 - 0.5151 & - 0.4500 & 0.7923 & 652.2641 \\\\
\\end{bmatrix}$$

**Extrinsic parameters with LM and distortion incorporated:**

$$\\left\\lbrack R\_{35}\|{\\overrightarrow{t}}\_{35} \\right\\rbrack\_{\\text{LM}} = \\begin{bmatrix}
0.9466 & - 0.0793 & - 0.4513 & - 62.8548 \\\\
 - 0.0659 & 0.9978 & - 0.0663 & - 92.6590 \\\\
 - 0.4534 & - 0.0496 & 0.9471 & 632.2045 \\\\
\\end{bmatrix}$$

**Camera pose for fixed imaged with measured ground-truth :**

The \[*R*<sub>11</sub>\|*t⃗*<sub>11</sub>\]for the fixed image 11 with
LLS and LM are given below. It can be seen that R is approximately
identity matrix with t representing the translation.

$$\\left\\lbrack R\_{11}\|{\\overrightarrow{t}}\_{11} \\right\\rbrack\_{\\text{LLS}} = \\begin{bmatrix}
0.9994 & - 0.0072 & 0.0342 & 19.5082 \\\\
0.0088 & 0.9989 & - 0.0469 & - 170.1235 \\\\
 - 0.0338 & 0.0472 & 0.9983 & 633.9512 \\\\
\\end{bmatrix}$$

$$\\left\\lbrack R\_{11}\|{\\overrightarrow{t}}\_{11} \\right\\rbrack\_{\\text{LM}} = \\begin{bmatrix}
0.9 & - 0.0082 & 0.0418 & - 78.9214 \\\\
0.0084 & 0.9999 & - 0.0568 & - 95.9668 \\\\
 - 0.0418 & 0.0569 & 0.9998 & 630.0702 \\\\
\\end{bmatrix}$$

**References:**

1.  Zhengyou Zhang. A flexible new technique for camera calibration.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    22:1330-1334, December 2000. MSR-TR-98-71, Updated March 25, 1999.
