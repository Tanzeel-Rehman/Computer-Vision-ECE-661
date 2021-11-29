**Image Classification using Gram matrices based textural features and
SVM classifier**

This homework performs image classification on 4 different classes using
the textural features extracted from Gram matrix representation combined
with the support vector machines (SVM) classifier. The performance of
the classification task on the test data was reported in terms of
confusion matrix and overall classification accuracy.

**Gram matrix feature extraction:**

The Gram matrix representation of image texture is actually inspired
from the deep learning. In order to construct a Gram matrix of an image,
it was convolved with the *C* different convolutional kernels having a
size of 3 × 3. Each convolutional kernel was generated randomly from
uniform distribution having a range of -1 to 1 with weight summing to 0.
After convolving the image with *C* kernels, we reduced the size of each
convolved image using linear interpolation to 16× 16 pixels. Each image
obtained by convolutional operation was considered as an output channel
(leading to *C* output channels), therefore, leading to an output image
having a size of 16 × 16 × *C* (In case of *C* = 10, the output image
will have a size of 16 × 16 × 10). Next, we vectorized, each channel
output, therefore representing each output channel by 256 elemental
vector. Finally, we took the dot product of all these output channels
leaving into a matrix having a size of *C* × *C*. This matrix is called
as Gram matrix and since it is symmetric in nature, therefore, we only
retained the upper triangular part including diagonal elements as the
feature vector to train the SVM classifier.

In this homework, we also used image center cropping rather than
resizing for Gram matrix extraction. We performed the center cropping
having a size of 96 × 96 pixels. After convolving these images with *C*
convolutional operators, we had an output image having a size of 96 × 96
× *C*. From here we proceed as indicated in the above paragraph to
extract the Gram matrix features.

The Gram matrix for the random training images are provided in the
figure below for image resizing (Figure 1) and image cropping (Figure
2), respectively. It can be seen that the image cropping generally
contained more clustered and distinct textural information, while for
the image resizing there is visually less distinction among instances of
different classes. compared to the image resizing. This is especially
true for the cloudy, shine and sunrise images. This could be due to the
fact that in the image resizing some information can be lost due to the
blurring effect caused by resizing.

**Results:**

<img src="media\image1.jpeg" style="width:3.16154in;height:3in" />

**(a)**

<img src="media\image2.jpeg" style="width:3.04651in;height:3in" />

**(b)**

<img src="media\image3.jpeg" style="width:3.0617in;height:3in" />

**(c)**

<img src="media\image4.jpeg" style="width:3.02278in;height:3in" />

> **(d)**

**Gram matrices for the randomly picked samples from the training data
for resized images at C = 70, (a) cloudy, (b) rain image (c) sunshine
image and (d) sunrise image.**

<img src="media\image5.jpeg" style="width:3.07653in;height:3in" />

**(a)**

<img src="media\image6.jpeg" style="width:3.20635in;height:3in" />

**(b)**

<img src="media\image7.jpeg" style="width:3.08616in;height:3in" />

**(c)**

<img src="media\image8.jpeg" style="width:3.03846in;height:3in" />

**(d)**

**Gram matrices for the randomly picked samples from the training data
for center cropped images at C = 12, (a) cloudy, (b) rain image (c)
sunshine image and (d) sunrise image.**

<img src="media\image9.png" style="width:5.91667in;height:4.7125in" />

**(a)**

<img src="media\image10.png" style="width:6.27083in;height:4.47917in" />

**(b)**

**Training and test accuracies against the C parameter of the Gram
Matrix (a) for resized images and (best validation accuracy of 50% at C
= 70) (b) for center cropped images (best validation accuracy of 50.61%
at C = 12).**

The overall accuracies for the image resizing and image cropping were
40% and 50%, respectively on the test dataset as can be seen from the
confusion matrices shown in Figure 4.

<img src="media\image11.png" style="width:5.51042in;height:4.28125in" />

**(a)**

<img src="media\image12.png" style="width:5.58333in;height:4.28125in" />

**(b)**

**Confusion matrix of test dataset obtained with Gram matrix features
and SVC classifier (a) for resized images with C = 70 (Total accuracy =
40%) and (b) for center cropped images with C = 12 (Total accuracy =
50%).**

**References:**

1.  Gbeminiyi Ajayi. Multi-class Weather Dataset for Image
    Classification. Available at
    <http://dx.doi.org/10.17632/4drtyfjtfy.1>.
