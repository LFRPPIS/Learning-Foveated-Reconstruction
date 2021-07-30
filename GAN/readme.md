Requirements
--

1. Python 3.3-3.7
2. CUDA 10.0
3. CuDNN 7.6
4. All required packages installed using the command ```pip install requirements.txt```

GAN prediction
--

To run the pretrained network, use one of the following commands. The results will be saved to the `results` directory.

1. LPIPS Ours, far periphery:

```
python run_gan.py predict -d lpips_dist_far -c pretrained/lpips_dist_far -t dataset/test_far
```

2. LPIPS Ours, near periphery:

```
python run_gan.py predict -d lpips_dist_near -c pretrained/lpips_dist_near -t dataset/test_near
```

3. LPIPS, Far periphery:

```
python run_gan.py predict -d lpips_real_far -c pretrained/lpips_real_far -t dataset/test_far
```

4. LPIPS, near periphery:

```
python run_gan.py predict -d lpips_real_near -c pretrained/lpips_real_near -t dataset/test_near
```

5. L2 Ours, far periphery:

```
python run_gan.py predict -d l2_dist_far -c pretrained/l2_dist_far -t dataset/test_far
```

6. L2 Ours, near periphery:

```
python run_gan.py predict -d l2_dist_near -c pretrained/l2_dist_near -t dataset/test_near
```

7. L2, far periphery:

```
python run_gan.py predict -d l2_real_far -c pretrained/l2_real_far -t dataset/test_far
```

8. L2, near periphery:

```
python run_gan.py predict -d l2_real_near -c pretrained/l2_real_near -t dataset/test_near
```

9. Laplacian Ours with peak at 0, far periphery:

```
python run_gan.py predict -d lapl_0_dist_far -c pretrained/lapl_0_dist_far -t dataset/test_far
```

10. Laplacian Ours with peak at 0, near periphery:

```
python run_gan.py predict -d lapl_0_dist_near -c pretrained/lapl_0_dist_near -t dataset/test_near
```

11. Laplacian with peak at 0, far periphery:

```
python run_gan.py predict -d lapl_0_real_far -c pretrained/lapl_0_real_far -t dataset/test_far
```

12. Laplacian with peak at 0, near periphery:

```
python run_gan.py predict -d lapl_0_real_near -c pretrained/lapl_0_real_near -t dataset/test_near
```

13. Laplacian Ours with peak at 3, far periphery:

```
python run_gan.py predict -d lapl_3_dist_far -c pretrained/lapl_3_dist_far -t dataset/test_far
```

14. Laplacian Ours with peak at 3, near periphery:

```
python run_gan.py predict -d lapl_3_dist_near -c pretrained/lapl_3_dist_near -t dataset/test_near
```

15. Laplacian with peak at 3, far periphery:

```
python run_gan.py predict -d lapl_3_real_far -c pretrained/lapl_3_real_far -t dataset/test_far
```

16. Laplacian with peak at 3, near periphery:

```
python run_gan.py predict -d lapl_3_real_near -c pretrained/lapl_3_real_near -t dataset/test_near
```

GAN Training
--

Use the following command to run the network training:

```
run_gan.py [-h] [-p [lpips, l2, lapl]] [-lp LAPLACIAN_PEAK] 
                              [-cv [8, 14]] [-g GROUND_TRUTH]
                              [-i INPUT] [-r REAL] [-d RESULTS] [-e EPOCHS]
                              [-t TEST_INPUT] [-b BATCH_SIZE]
                              [predict, train]
```

Positional arguments:

1. `[predict, train]` specify type of action to perform on a network

Optional arguments:

1. `-p [lpips, l2, lapl], --perceptual_loss_type [lpips, l2, lapl]` used loss metric for the generator
2. `-lp LAPLACIAN_PEAK --laplacian_pyramid_peak_position LAPLACIAN_PEAK` Laplacian pyramid level with the highest
   weight (only with Laplacian loss)
3. `-g GROUND_TRUTH, --ground_truth GROUND_TRUTH` the images folder used as a ground truth for the generator
4. `-i INPUT, --input INPUT` the images folder used as an input for the generator
5. `-r REAL, --real REAL`  the images folder treated as 'real' by the discriminator
6. `-d RESULTS, --directory RESULTS` the folder in which all generated test images would be saved
7. `-e EPOCHS, --epochs EPOCHS` number of epochs for training
8. `-t TEST_INPUT, --test_input TEST_INPUT` the images folder used for testing the network each epoch
   9 `-b BATCH_SIZE, --batch_size BATCH_SIZE` number of images put in a batch during training

Images used as a training dataset must have a size of 256x256.

To train the network, custom dataset is required. It can be generated using the Interpolation and Image synthesis
script. Additionally, we provide the LPIPS pretrained network from https://github.com/richzhang/PerceptualSimilarity in
a zip file in the GAN/ directory, which needs to be extracted inplace. Examples of available training procedures:

1. LPIPS Ours, far periphery:

```
python run_gan.py train -p lpips -g dataset/ground_truth -i dataset/train_far -r dataset/distorted_far -d train_lpips_distorted_far -t dataset/test_far -b 8 -e 20
```

2. LPIPS Ours, near periphery:

```
python run_gan.py train -p lpips -g dataset/ground_truth -i dataset/train_near -r dataset/distorted_near -d train_lpips_distorted_near -t dataset/test_near -b 8 -e 20
```

3. LPIPS, far periphery:

```
python run_gan.py train -p lpips -g dataset/ground_truth -i dataset/train_far -r dataset/ground_truth -d train_lpips_standard_far -t dataset/test_far -b 8 -e 20
```

4. LPIPS, near periphery:

```
python run_gan.py train -p lpips -g dataset/ground_truth -i dataset/train_near -r dataset/ground_truth -d train_lpips_standard_near -t dataset/test_near -b 8 -e 20
```

5. L2 Ours, far periphery:

```
python run_gan.py train -p l2 -g dataset/ground_truth -i dataset/train_far -r dataset/distorted_far -d train_l2_distorted_far -t dataset/test_far -b 8 -e 20
```

6. L2 Ours, near periphery:

```
python run_gan.py train -p l2 -g dataset/ground_truth -i dataset/train_near -r dataset/distorted_near -d train_l2_distorted_near -t dataset/test_near -b 8 -e 20
```

7. L2, far periphery:

```
python run_gan.py train -p l2 -g dataset/ground_truth -i dataset/train_far -r dataset/ground_truth -d train_l2_standard_far -t dataset/test_far -b 8 -e 20
```

8. L2, near periphery:

```
python run_gan.py train -p l2 -g dataset/ground_truth -i dataset/train_near -r dataset/ground_truth -d train_l2_standard_near -t dataset/test_near -b 8 -e 20
```

9. Laplacian Ours with peak at 0, far periphery:

```
python run_gan.py train -p lapl -lp 0 -g dataset/ground_truth -i dataset/train_far -r dataset/distorted_far -d train_lapl_0_distorted_far -t dataset/test_far -b 8 -e 20
```

10. Laplacian Ours with peak at 0, near periphery:

```
python run_gan.py train -p lapl -lp 0 -g dataset/ground_truth -i dataset/train_near -r dataset/distorted_near -d train_lapl_0_distorted_near -t dataset/test_near -b 8 -e 20
```

11. Laplacian with peak at 0, far periphery:

```
python run_gan.py train -p lapl -lp 0 -g dataset/ground_truth -i dataset/train_far -r dataset/ground_truth -d train_lapl_0_standard_far -t dataset/test_far -b 8 -e 20
```

12. Laplacian with peak at 0, near periphery:

```
python run_gan.py train -p lapl -lp 0 -g dataset/ground_truth -i dataset/train_near -r dataset/ground_truth -d train_lapl_0_standard_near -t dataset/test_near -b 8 -e 20
```

13. Laplacian Ours with peak at 3, far periphery:

```
python run_gan.py train -p lapl -lp 3 -g dataset/ground_truth -i dataset/train_far -r dataset/distorted_far -d train_lapl_3_distorted_far -t dataset/test_far -b 8 -e 20
```

14. Laplacian Ours with peak at 3, near periphery:

```
python run_gan.py train -p lapl -lp 3 -g dataset/ground_truth -i dataset/train_near -r dataset/distorted_near -d train_lapl_3_distorted_near -t dataset/test_near -b 8 -e 20
```

15. Laplacian with peak at 3, far periphery:

```
python run_gan.py train -p lapl -lp 3 -g dataset/ground_truth -i dataset/train_far -r dataset/ground_truth -d train_lapl_3_standard_far -t dataset/test_far -b 8 -e 20
```

16. Laplacian with peak at 3, near periphery:

```
python run_gan.py train -p lapl -lp 3 -g dataset/ground_truth -i dataset/train_near -r dataset/ground_truth -d train_lapl_3_standard_near -t dataset/test_near -b 8 -e 20
```

Interpolation
--

Used for generating subsampled interpolated images used as an input to the GAN network. Usage:

```
generate_interpolated_data.py [-h] [-s SAMPLING] [-o OUTPUT] [-i INPUT] [-nu]
```

Optional arguments:

1. `-s SAMPLING, --sampling SAMPLING` sampling rate of the synthesis
2. `-o OUTPUT, --output OUTPUT` the folder for generated images
3. `-i INPUT, --input INPUT` the folder with input images
4. `-nu --non-uniform` use non-uniform mask instead of a uniform

Example usage:

```
python generate_interpolated_data.py -s 0.007 -o dataset/interpolated_0.007 -i dataset/ground_truth
```

Image synthesis
--

Performes image synthesis using a specified number of guided samples. Usage:

```
texture_synthesis.py [-h] [-s SAMPLING] [-o OUTPUT] [-i INPUT]
```

Optional arguments:

1. `-s SAMPLING, --sampling SAMPLING` guided sampling rate of the synthesis
2. `-o OUTPUT, --output OUTPUT` the folder for generated images
3. `-i INPUT, --input INPUT` the folder with input images

Example usage:

```
python texture_synthesis.py -s 0.007 -o dataset/synthesized_0.007 -i dataset/ground_truth
```

Mean detection computation
--

Computes the predicted probability of detecting differences between original and distorted image, according to our
calibrated VGG metric.

```
predict_detection.py [-h] [-r REAL] [-d DISTORTED]
```

Optional arguments:

1. `-r REAL, --real REAL` path to the original image
2. `-d DISTORTED, --distorted DISTORTED` path to the compared image

Example usage:

```
python predict_detection.py -r original_image.png -d distorted_image.png
```