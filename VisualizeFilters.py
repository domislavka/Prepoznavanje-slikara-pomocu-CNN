import numpy as np
from time import time
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img

# https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

def layerFilters(model, layer_name, img_width, img_height, num_filters):

    """
    Funkcija za vizualizaciju filtera mreze.
    Filtere primjenjene na sliku sprema u datoteku.
    """

    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    kept_filters = []
    input_img = model.input
    filter_img = load_img('ImageForFilter.jpg')

    for filter_index in range(num_filters):
        # print('Processing filter %d' % filter_index)

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        input_img_data = img_to_array(filter_img)
        input_img_data = input_img_data.reshape(1, img_width, img_height, 3)

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            # print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))


    # we will try to stitch the best 9 filters on a 3 x 3 grid.
    n = 3
    
    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 9 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 4 x 4 filters of size 224 x 224, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    save_img('stitched_filters_' + layer_name + '_%dx%d.png' % (n, n), stitched_filters)
    print('Filters saved to file ' + 'stitched_filters_' + layer_name + '_%dx%d.png' % (n, n))

    # show image in window
    stitched_filters.show()



