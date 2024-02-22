import numpy as np


def tile_images(img_nhwc):
    img_nhwc = np.asarray(img_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(
        list(img_nhwc)
        + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = img_nhwc.reshape(new_height, new_width, height, width, n_channels)
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape(new_height * height, new_width * width, n_channels)
    return out_image
