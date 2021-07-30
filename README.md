# Learning-Foveated-Reconstruction

This repository is provided as a supplementary material to the "Learning Foveated Reconstruction to Preserve Perceived Image Statistics" paper. It includes the following directories:
1. `GAN` - allows to generate the images based on all the methods explained in the paper in Section 5. We provide the pretrained GAN along with test images. We also provide an option of computing predicted detection rate of distortions using our calibrated VGG metric.
2. `demo` - shows a detailed comparison of images generated using different training procedures, explained in the paper Section 5. The demonstration can be started by launching `reconstructions.html` in a browser. It is fully offline and can be started without internet connection. We have tested it using Firefox and Chrome.
