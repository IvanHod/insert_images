import numpy as np
from cv2 import cv2
from src import transformation, parsing
from skimage import util


def calc_mask(pic, red_mask):
    mask = np.ones((pic.shape[0], pic.shape[1])).astype(bool)

    mask[red_mask] = False
    mask[(pic == [0, 0, 0]).all(axis=2)] = False

    return mask


def _insert_pic(img, pic, shift_outside, shift_inside: int, config: dict, img_index=None,
                img_source_mask=None):
    h, w, _ = pic.shape
    shift = shift_outside + np.array([0, shift_inside])

    red_mask = (img_source_mask[:, :, 2] > 240) & (img_source_mask[:, :, 0] < 10)
    red_mask_path = red_mask[shift[0]: shift[0] + h, shift[1]: shift[1] + w]
    mask = ~calc_mask(pic, red_mask_path)

    img_path = img[shift[0]: shift[0] + h, shift[1]: shift[1] + w, :]

    green_threshold = config.get('green_threshold')
    if img_index in config.get('green_filter', []) and green_threshold is not None:
        mask[img_path.mean(axis=2) < green_threshold] = True

    pic[mask] = img_path[mask]

    coef = np.ones((h, w)) * 2
    coef[~mask] = 0
    coef_big = np.ones((h + 2, w + 2)) * 2
    coef_big[1:-1, 1:-1] = coef

    for i in range(10):
        coef_big[1:-1, 1:-1] = util.view_as_windows(coef_big, (3, 3), 1).mean(axis=(2, 3))

    coef_big = np.clip(coef_big, 0, 1).reshape((h + 2, w + 2, 1))
    coef = coef_big[1:-1, 1:-1]

    pic = ((pic * (1 - coef)) + (img_path * coef)).astype('uint8')
    img[shift[0]: shift[0] + pic.shape[0], shift[1]: shift[1] + pic.shape[1], :] = pic

    return img


def insert_pic(img, pic, box, config, to_resize=True, linear=True, img_index=None,
               img_source_mask=None):
    box_h, box_w = box['left'][-1, 0] - box['left'][0, 0], box['bottom'][-1, 1] - box['bottom'][0, 1]

    pic_resized = transformation.to_resize_pic(pic, box) if to_resize else pic

    warped_cut, warped, img_width_new = transformation.do_transform(pic_resized, box, kind='linear' if linear else 'affine')

    shift_inside = int((box_w - pic_resized.shape[1]) / 2)
    shift = [
        min(box['left'][:, 0].min(), box['right'][:, 0].min()),
        box['left'][:, 1].min()  # + shift_inside
    ]

    img = _insert_pic(img, warped, shift, shift_inside, config, img_index=img_index,
                      img_source_mask=img_source_mask)

    return img


def insert_right_pic(img, pic, box, config):
    pic_h, pic_w, _ = pic.shape
    box_h, box_w = box['left'][-1, 0] - box['left'][0, 0], box['bottom'][-1, 1] - box['bottom'][0, 1]

    if box_w < box_h:
        pic = pic[:, :int(pic_w * (box_w / box_h))]

    return insert_pic(img, pic, box, config=config, to_resize=True, linear=True)


def insert_pictures_into_frame(frame, boxes, images_config, index: int):
    img_work = frame.copy()

    config = images_config[index]

    if boxes['left'] is not None:
        img_work = insert_pic(img_work, config['images'][0], config=config, box=boxes['left'],
                              linear=True, img_index=0)
    if boxes['right'] is not None:
        img_work = insert_right_pic(img_work, config['images'][-1], config=config, box=boxes['right'])

    indices = list(filter(lambda i: config['dark'][i] > 0, range(1, 8)))
    images = list(map(lambda v: config['images'][v], indices))
    for i, pic, box in zip(indices, images, boxes['middle']):
        img_work = insert_pic(img_work, pic, box=box, config=config, linear=False, img_index=i)

    if config['right-path']:
        img_work = img_work[:, :-20, :]

    # plt.imsave('out.png', cv2.cvtColor(img_work, cv2.COLOR_RGB2BGR))
    # imshow(img_work[:600, :])
    # insert_boxes(img_work, boxes)
    return img_work


def insert_pictures_into_video(img_source_mask, images_config, index: int, stop_frame=None):
    boxes = parsing.create_boxes(img_source_mask, config=images_config[index])

    filename = f'video-{index}'
    vidcap = cv2.VideoCapture(f'data/{filename}.MOV')

    frames = []
    success, iteration = True, 0
    while success:
        print(f'Iteration {iteration}')
        success, frame = vidcap.read()
        if not success:
            break

        try:
            frame_out = insert_pictures_into_frame(frame, boxes, images_config, index=index)

            frames.append(frame_out)

        except Exception as e:
            print('error', e)
            break

        iteration += 1
        if stop_frame is not None and iteration > stop_frame:
            break

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(f'data/out/{filename}.mov', fourcc, 20.0, (w, h))

    for i in range(len(frames)):
        out.write(frames[i])
    out.release()
