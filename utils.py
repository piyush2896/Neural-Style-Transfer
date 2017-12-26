import numpy as np
import scipy
import scipy.io
import scipy.misc


vgg_mean = np.array([123.68, 116.779, 103.939])

def generate_noisy_image(content_img, img_width=300,
                         img_height=300, channels=3, noise_ratio=0.6):
    noise_image = np.random.uniform(-20, 20, (1, img_height, img_width, channels)).astype('float32')
    input_image = noise_image * noise_ratio + content_img * (1-noise_ratio)
    return input_image


def save_image(img, path):
    img = img + vgg_mean
    img = np.clip(img[0], 0, 255).astype('uint8')
    scipy.misc.imsave(path, img)


def load_image(path, size):
    img = scipy.misc.imread(path)
    img = img - vgg_mean
    img = scipy.misc.imresize(img, size=size)
    return np.expand_dims(img, axis=0)