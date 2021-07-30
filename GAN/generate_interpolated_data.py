from glob import glob
import numpy as np
import scipy.interpolate
import BlueNoise
import cv2
from argparse import ArgumentParser


def interpolate(sparse_images, masks, method='linear'):
    width = sparse_images.shape[0]
    height = sparse_images.shape[1]
    coord_x, coord_y = np.meshgrid(range(height), range(width))

    mask_x = coord_x[masks == 1].reshape((-1, 1))
    mask_y = coord_y[masks == 1].reshape((-1, 1))

    val = sparse_images
    grid_x = coord_x
    grid_y = coord_y
    points = np.concatenate([mask_x, mask_y], axis=1)
    values = val[points[:, 1], points[:, 0], :]
    grid = scipy.interpolate.griddata(points, values, (grid_x, grid_y), method=method)
    grid = np.clip(grid, 0, 1)

    return grid


def void_and_cluster(mask, std, patch_size):
    height = mask.shape[0]
    width = mask.shape[1]

    initial_binary_pattern = [None] * (height // patch_size) * (width // patch_size)
    k = 0
    for i in range(height // patch_size):
        for j in range(width // patch_size):
            initial_binary_pattern[k] = mask[i * 256: i * 256 + 256, j * 256: j * 256 + 256]
            k += 1

    for i in range(len(initial_binary_pattern)):
        if np.count_nonzero(initial_binary_pattern[i]) == 0:
            continue
        revert = False
        if np.count_nonzero(initial_binary_pattern[i]) * 2 == patch_size * patch_size:
            initial_binary_pattern[i][0, 0] = 1.0 - initial_binary_pattern[i][0, 0]
        if np.count_nonzero(initial_binary_pattern[i]) * 2 >= patch_size * patch_size:
            initial_binary_pattern[i] = 1.0 - initial_binary_pattern[i]
            revert = True

        initial_binary_pattern[i] = BlueNoise.GetVoidAndClusterBlueNoise(initial_binary_pattern[i], std)
        if revert:
            initial_binary_pattern[i] = 1.0 - initial_binary_pattern[i]

    k = 0
    for i in range(height // patch_size):
        for j in range(width // patch_size):
            mask[i * 256: i * 256 + 256, j * 256: j * 256 + 256] = initial_binary_pattern[k]
            k += 1

    return mask


def generate_uniform_mask(size, samples_fraction):
    total_samples = float(size[0] * size[1])
    used_samples = int(np.round(float(total_samples) * samples_fraction))
    total_samples = int(total_samples)
    not_used_samples = total_samples - used_samples
    not_used_samples = np.zeros((not_used_samples))
    used_samples = np.ones((used_samples))
    mask = np.concatenate((not_used_samples, used_samples), 0)

    np.random.shuffle(mask)
    mask = np.reshape(mask, [size[0], size[1]])
    return mask


def generate_non_uniform_mask(size, samples_fraction):
    total_samples = int(samples_fraction * size[0] * size[1])
    x, y = np.meshgrid(range(size[1]), range(size[0]))
    x = abs(x - (size[1] - 1) / 2)
    y = abs(y - (size[0] - 1) / 2)
    distances = (x ** 2 + y ** 2) ** 0.5
    distances = 1 - distances / np.max(distances)
    distances = distances * total_samples / np.sum(distances)
    probabilities = np.random.uniform(0.0, 1.0, (size[0], size[1]))
    mask = np.where(probabilities > distances, 0, 1)
    return mask


def save_image_data(input_path, output_path, size, sampling_rate, uniform):
    filename = input_path.split('\\')[-1]
    image = cv2.imread(input_path)
    image = image.astype(float) / 255.0

    enlarged_mask_size = [size[0], size[1]]
    enlarged_mask_size[0] = ((size[0] + 255) // 256) * 256
    enlarged_mask_size[1] = ((size[1] + 255) // 256) * 256

    if uniform:
        mask = generate_uniform_mask(size=enlarged_mask_size, samples_fraction=sampling_rate)
    else:
        mask = generate_non_uniform_mask(size=size, samples_fraction=sampling_rate)
    mask = void_and_cluster(mask, 5.0, 256)
    mask = mask[0: size[0], 0: size[1]]
    mask[0, 0] = 1
    mask[size[0] - 1, 0] = 1
    mask[0, size[1] - 1] = 1
    mask[size[0] - 1, size[1] - 1] = 1
    mask = np.expand_dims(mask, axis=2)
    sparse = image * mask

    interpolated = (interpolate(sparse, np.squeeze(mask)) * 255).astype(np.uint8)

    cv2.imwrite(f'{output_path}/{filename}_interpolated.png', interpolated)

    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(f'{output_path}/{filename}_mask.png', mask)

    return filename


def load_images(input_directory, output_directory, image_size, sampling_rate, uniform):
    all_image_paths = glob(f'{input_directory}/*.png')
    for image_path in all_image_paths:
        save_image_data(image_path, output_directory, size=image_size, sampling_rate=sampling_rate, uniform=uniform)


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--sampling", dest="sampling",
                        help="sampling rate of the synthesis")
    parser.add_argument("-o", "--output", dest="output",
                        help="the folder for generated images")
    parser.add_argument("-i", "--input", dest="input",
                        help="the folder with input images")
    parser.add_argument("-nu", "--non-uniform", dest="non_uniform", action="store_true",
                        help="use non-uniform mask instead of a uniform")

    args = parser.parse_args()

    test_image_path = glob(f'{args.input}/*.png')[0]
    test_image = cv2.imread(test_image_path)
    image_size = (test_image.shape[0], test_image.shape[1])

    load_images(args.input, args.output, image_size=image_size, sampling_rate=float(args.sampling),
                uniform=not args.non_uniform)


if __name__ == '__main__':
    main()
