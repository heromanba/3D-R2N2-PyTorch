import numpy as np
from lib.config import cfg

from PIL import Image


def image_transform(img, crop_x, crop_y, crop_loc=None, color_tint=None):
    """
    Takes numpy.array img
    """

    # Slight translation
    if cfg.TRAIN.RANDOM_CROP and not crop_loc:
        crop_loc = [np.random.randint(0, crop_y), np.random.randint(0, crop_x)]

    if crop_loc:
        cr, cc = crop_loc
        height, width, _ = img.shape
        img_h = height - crop_y
        img_w = width - crop_x
        img = img[cr:cr + img_h, cc:cc + img_w]
        # depth = depth[cr:cr+img_h, cc:cc+img_w]

    if cfg.TRAIN.FLIP and np.random.rand() > 0.5:
        img = img[:, ::-1, ...]

    return img


def crop_center(im, new_height, new_width):
    height = im.shape[0]  # Get dimensions
    width = im.shape[1]
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return im[top:bottom, left:right]


def add_random_color_background(im, color_range):
    r, g, b = [np.random.randint(color_range[i][0], color_range[i][1] + 1) for i in range(3)]

    if isinstance(im, Image.Image):
        im = np.array(im)

    if im.shape[2] > 3:
        # If the image has the alpha channel, add the background
        alpha = (np.expand_dims(im[:, :, 3], axis=2) == 0).astype(np.float)
        im = im[:, :, :3]
        bg_color = np.array([[[r, g, b]]])
        im = alpha * bg_color + (1 - alpha) * im

    return im


def preprocess_img(im, train=True):
    # add random background
    im = add_random_color_background(im, cfg.TRAIN.NO_BG_COLOR_RANGE if train else
                                     cfg.TEST.NO_BG_COLOR_RANGE)

    # If the image has alpha channel, remove it.
    im_rgb = np.array(im)[:, :, :3].astype(np.float32)
    if train:
        t_im = image_transform(im_rgb, cfg.TRAIN.PAD_X, cfg.TRAIN.PAD_Y)
    else:
        t_im = crop_center(im_rgb, cfg.CONST.IMG_H, cfg.CONST.IMG_W)

    # Scale image
    t_im = t_im / 255.

    return t_im

def test(fn):
    import matplotlib.pyplot as plt
    cfg.TRAIN.RANDOM_CROP = True
    im = Image.open(fn)
    im = np.asarray(im)[:, :, :4]
    print(np.asarray(im).shape)
    imt = preprocess_img(im, train=False)
    print(imt.shape)
    plt.imshow(imt)
    plt.show()
    
    
"""two more functions to do data augumentation"""
def crop_randomly(im, new_height, new_width):
    height = im.shape[0]
    width = im.shape[1]
    try:
        assert((new_height <= height) and (new_width <= width))
    except AssertionError:
        raise Exception("both new_height <= height and width <= new_width must be satisfied")
    left = np.random.randint(0, width - new_width)
    top = np.random.randint(0, height - new_height)
    right = left + new_height
    bottom = top + new_width
    return im[top:bottom, left:right]


#add a specific image as the background
def add_bgd_image(im, bgd_im):
    if isinstance(im, Image.Image):
        im = np.asarray(im)
    if isinstance(bgd_im, Image.Image):
        bgd_im = np.asarray(bgd_im)
    #trigger an error if either im or bgd_im is not an ndarray
    assert(isinstance(im, np.ndarray) and isinstance(bgd_im, np.ndarray))
    try:
        assert((im.shape[2] == 4) and (bgd_im.shape[2] == 3))
        #crop bgd_im to the same shape as im
        bgd_im = crop_center(bgd_im, im.shape[0], im.shape[1])
        #find points in im which have no background
        alpha = (np.expand_dims(im[:,:,3], axis=2) ==0 ).astype(np.float)
        #drop the alpha channel
        im = im[:, :, :3]
        #add the bgd_im to the background of im
        im = alpha * bgd_im + (1 - alpha) * im
    except AssertionError:
        raise Exception("argument \"im\" must have an alpha channel or \
                                  \"bgd_im\" must have no aplha channel")
    return im


if __name__ == '__main__':
    #test("/Users/wangchu/Desktop/shapeNet_rendering_chair03.png")
    import scipy.misc
    
    bgd_im = Image.open("/Users/wangchu/Desktop/mountain.jpg")
    bgd_im = np.asarray(bgd_im)
    bgd_im = crop_randomly(bgd_im, 137, 137)
    
    im = Image.open("/Users/wangchu/Desktop/00.png")
    im = np.asarray(im)
    
    im = add_bgd_image(im, bgd_im)
    scipy.misc.imsave("/Users/wangchu/Desktop/00_with_bgd.jpg", im)
    
    import matplotlib.pyplot as plt
    plt.imshow(im)
    plt.show
