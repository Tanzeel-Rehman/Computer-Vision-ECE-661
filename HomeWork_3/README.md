**Removing projective and affine distortions from images**

This homework uses three different methods for removing the projective
and affine distortions from the images. These methods return the
homography matrices, which can then be inverted and applied on the
real-world distorted images to eliminate the distortions.

1.  The first method uses point-to-point correspondences (similar to the
    homework 2) between image pixel coordinates of the images and the
    points obtained from the given physical measurements for computing
    the homography matrix.

2.  The second method is a two-step approach in which we first remove
    the projective distortion using the Vanishing line approach (Lecture
    4). In the second step, we removed the affine distortion by using
    the angle-to-angle correspondence (*Cosθ* expression) with *θ* = 90
    degrees i.e., orthogonal lines.

3.  The third method is a one-step approach in which both projective and
    affine distortion are removed in one phase.

**Results:**

For complete results see the results provided in pdf format.

<img src="media\image1.JPG" style="width:6.5in;height:4.875in" alt="A car parked in front of a building Description automatically generated" />

**Input Image 1. The points PQSR were used for removing the distorting
using all three methods**

<img src="media\image2.jpeg" style="width:6.5in;height:7.02014in" />

**Distortion removal from input image 1 using the point-to-point
correspondence method**

<img src="media\image3.jpeg" style="width:3.25302in;height:8.41667in" />

**Projective distortion removal from input image 1 using vanishing lines
method (1<sup>st</sup> of 2-steps)**

<img src="media\image4.jpeg" style="width:6.5in;height:6.13819in" />

**Affine distortion removal from input image 1 using angle-to-angle
correspondence (2<sup>nd</sup> of 2-steps).**

<img src="media\image5.jpeg" style="width:6.5in;height:6.26875in" />

**Removing both projective and affine distortion from input image 1
using one-step method.**

<img src="media\image6.jpeg" style="width:6.27885in;height:3.53118in" />

**Input Image 2. The points PQSR were used for removing the distorting
using all three methods**

<img src="media\image7.jpeg" style="width:6.5in;height:3.91389in" />

**Distortion removal from input image 2 using the point-to-point
correspondence method**

<img src="media\image8.jpeg" style="width:6.5in;height:3.23056in" />

**Projective distortion removal from input image 2 using vanishing lines
method (1<sup>st</sup> of 2-steps).**

<img src="media\image9.jpeg" style="width:6.5in;height:5.325in" />

**Affine distortion removal from input image 2 using angle-to-angle
correspondence (2<sup>nd</sup> of 2-steps).**

<img src="media\image10.jpeg" style="width:6.5in;height:5.325in" />

**Removing both projective and affine distortion from input image 2
using one-step method.**
