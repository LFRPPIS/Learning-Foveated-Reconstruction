import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import GanModel
from argparse import ArgumentParser
import time


def main():
    parser = ArgumentParser()
    parser.add_argument("prediction_type", metavar="[predict, train]",
                        help="specify type of action to perform on a network")

    parser.add_argument("-p", "--perceptual_loss_type", dest="loss_type", default="lpips",
                        help="used loss metric for the generator", metavar="[lpips, l2, lapl]")
    parser.add_argument("-lp", "--laplacian_pyramid_peak_position", dest="laplacian_peak", default=0, type=int,
                        help="laplacian pyramid level with the highest weight (only with laplacian loss)")
    parser.add_argument("-g", "--ground_truth", dest="ground_truth", default="",
                        help="the images folder used as a ground truth for the generator")
    parser.add_argument("-i", "--input", dest="input", default="",
                        help="the images folder used as an input for the generator")
    parser.add_argument("-r", "--real", dest="real", default="",
                        help="the images folder treated as 'real' by the discriminator")
    parser.add_argument("-d", "--directory", dest="results", default="",
                        help="the folder name for all generated test images, logs and checkpoints")
    parser.add_argument("-c", "--checkpoints", dest="checkpoints",
                        help="the folder from which to load the checkpoints")
    parser.add_argument("-e", "--epochs", dest="epochs", default=1, type=int,
                        help="number of epochs for training")
    parser.add_argument("-t", "--test_input", dest="test_input",
                        help="the images folder used for testing the network each epoch")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=1, type=int,
                        help="number of images put in a batch during training")

    args = parser.parse_args()
    gan_model = GanModel.GanModel(args=args)

    if args.prediction_type == "predict":
        gan_model.predict()
    else:
        gan_model.train()


if __name__ == "__main__":
    main()
