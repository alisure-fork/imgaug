import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa


"""
https://imgaug.readthedocs.io/en/latest/source/augmenters.html
https://imgaug.readthedocs.io/en/latest/source/api.html


1.
    Sequential

2.
    SomeOf
    OneOf
    Sometimes

3.
    WithColorspace
    ChangeColorspace
    Grayscale
    WithChannels

4.
    Lambda
    AssertLambda
    AssertShape

5.
    Resize
    CropAndPad
    Pad
    Crop
    Fliplr
    Flipud

6.
    Superpixels

7.
    GaussianBlur
    AverageBlur
    MedianBlur

8.
    Convolve
    Sharpen
    Emboss
    EdgeDetect
    DirectedEdgeDetect

9.
    Add
    AddElementwise
    AdditiveGaussianNoise
    Multiply
    MultiplyElementwise

10.
    Dropout
    CoarseDropout

11.
    Invert

12.
    ContrastNormalization

13.
    Affine(仿射变换)
    PiecewiseAffine
    ElasticTransformation(弹性变换)
"""


ia.seed(1)

image = ia.quokka(size=(256, 256))

seq = iaa.Sequential([iaa.Affine(translate_px={"x": -40}),
                      iaa.AdditiveGaussianNoise(scale=0.1*255)], random_order=False)

seq2 = iaa.Sequential([iaa.Affine(translate_px={"x": -40}),
                       iaa.AdditiveGaussianNoise(scale=0.1*255)], random_order=True)

images_aug = [seq.augment_image(image) for _ in range(8)]
images_aug2 = [seq2.augment_image(image) for _ in range(8)]
ia.imshow(ia.draw_grid(images_aug + images_aug2, cols=8))

aug = iaa.SomeOf(2, [
    iaa.Affine(rotate=45),
    iaa.AdditiveGaussianNoise(scale=0.2*255),
    iaa.Add(50, per_channel=True),
    iaa.Sharpen(alpha=0.5)
])
