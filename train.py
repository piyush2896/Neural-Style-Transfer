import tensorflow as tf
from vgg16 import vgg16
import numpy as np
import argparse
import sys
from utils import *
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser()
parser.add_argument("--cnt", metavar="cnt_img", help="Content Image for style transfer")
parser.add_argument("--stl", metavar="stl_img", help="Style Image for style transfer")
parser.add_argument("--size", metavar="size", nargs='+', type=int,
                    help="Size of output image [height, width]")
parser.add_argument("--niters", metavar="n_iters", help="Number of iterations to run for", type=int)
args = parser.parse_args()


STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_content_cost(a_content, a_generated):
    m , height, width, channels = a_generated.get_shape().as_list()
    shape = (height * width, channels, -1)
    cnt_unrolled = tf.transpose(tf.reshape(a_content, shape))
    gen_unrolled = tf.transpose(tf.reshape(a_generated, shape))

    den = 4 * shape[0] * shape[1]
    content_loss = tf.reduce_sum(tf.square(tf.subtract(cnt_unrolled, gen_unrolled))) / den
    return content_loss


def get_gram_matrix(A):
    return tf.matmul(A, tf.transpose(A))


def compute_layer_style_cost(a_style, a_generated):
    m, height, width, channels = a_generated.get_shape().as_list()
    shape = (height * width, channels)
    stl_unrolled = tf.transpose(tf.reshape(a_style, shape))
    gen_unrolled = tf.transpose(tf.reshape(a_generated, shape))

    stl_gram = get_gram_matrix(stl_unrolled)
    gen_gram = get_gram_matrix(gen_unrolled)

    den = 4 * (height * width) ** 2 * channels ** 2
    layer_style_cost = tf.reduce_sum(tf.square(tf.subtract(stl_gram, gen_gram))) / den
    return layer_style_cost


def compute_style_cost(sess, model, STYLE_LAYERS):
    style_cost = 0
    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_style = sess.run(out)
        a_generated = out
        layer_style_cost = compute_layer_style_cost(a_style, a_generated)
        style_cost += coeff * layer_style_cost
    return style_cost


def total_cost(content_cost, style_cost, alpha=10, beta=40):
    return alpha * content_cost + beta * style_cost


def train(args):
    model, params = vgg16(is_input_trainable=True, input_shape=[1, args.size[0], args.size[1], 3])

    content_img = load_image(args.cnt, size=args.size)
    style_img = load_image(args.stl, size=args.size)
    input_img = generate_noisy_image(content_img, img_width=args.size[1], img_height=args.size[0])

    optimizer = tf.train.AdamOptimizer(5.0)

    with tf.Session() as sess:
        sess.run(model['input'].assign(content_img))
        out = model['conv4_2']
        a_content = sess.run(out)
        a_generated = out
        content_cost = compute_content_cost(a_content, a_generated)

        sess.run(model['input'].assign(style_img))
        style_cost = compute_style_cost(sess, model, STYLE_LAYERS)

        cost = total_cost(content_cost, style_cost)

        train_step = optimizer.minimize(cost)

        sess.run(tf.global_variables_initializer())

        sess.run(model['input'].assign(input_img))

        for i in range(args.niters):
            sess.run(train_step)
            generated_img = sess.run(model['input'])

            if i % 20 == 0:
                c, c_cost, s_cost = sess.run([cost, content_cost, style_cost])
                print('Iteration ' + str(i) + ':')
                print('Total cost: {}\tContent Cost: {}\tStyle Cost: {}'.format(c, c_cost, s_cost))

                save_image(generated_img, './output/'+str(i)+'.png')
        save_image(generated_img, './output/generated_img.jpg')


if args.cnt == None:
    sys.exit("Path to Content Image Not Available")


if args.stl == None:
    sys.exit("Path to Style Image Not Available")


if args.size == None:
    args.size = [300, 300]

if args.niters == None:
    args.niters = 200

train(args)