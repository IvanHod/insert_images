from typing import Tuple

import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt

from src import transformation, parsing, tools
from skimage import util


def calc_mask(pic, red_mask):
    mask = np.ones((pic.shape[0], pic.shape[1])).astype(bool)

    mask[red_mask] = False
    mask[(pic == [0, 0, 0]).all(axis=2)] = False

    return mask


def _insert_pic(img, pic, shift_outside, shift_inside: int, config: dict, img_index=None,
                img_source_mask=None, smooth_count=10):
    h, w, _ = pic.shape
    shift = shift_outside + np.array([0, shift_inside])

    # mask to exclude the pixels of picture
    red_mask = (img_source_mask[:, :, 2] > 240) & (img_source_mask[:, :, 0] < 10)
    red_mask_path = red_mask[shift[0]: shift[0] + h, shift[1]: shift[1] + w]
    mask = ~calc_mask(pic, red_mask_path)

    # to cut from the image the piece corresponding to the picture
    img_path = img[shift[0]: shift[0] + h, shift[1]: shift[1] + w, :]

    # to add to the mask of excluding of the leaves from the trees
    green_threshold = config.get('green_threshold')
    if img_index in config.get('green_filter', []) and green_threshold is not None:
        mask[img_path.mean(axis=2) < green_threshold] = True

    # to cut leaves and objects from the red mask
    pic[mask] = img_path[mask]

    # to create the coefficients to smooth the borders
    coef = np.ones((h, w)) * 2
    coef[~mask] = 0
    # to add zeros on the border for the smoothing
    coef_big = np.ones((h + 2, w + 2)) * 2
    coef_big[1:-1, 1:-1] = coef

    # to smooth the borders from the zero
    for i in range(smooth_count):
        coef_big[1:-1, 1:-1] = util.view_as_windows(coef_big, (3, 3), 1).mean(axis=(2, 3))

    # to clip values less than 0 and more than 1
    coef_big = np.clip(coef_big, 0, 1).reshape((h + 2, w + 2, 1))
    coef = coef_big[1:-1, 1:-1]

    # inserting the picture to the image
    pic = ((pic * (1 - coef)) + (img_path * coef)).astype('uint8')
    img[shift[0]: shift[0] + pic.shape[0], shift[1]: shift[1] + pic.shape[1], :] = pic

    return img


def insert_pic(img, pic, box, config, to_resize=True, linear=True, img_index=None,
               img_source_mask=None, smooth_count=10):

    box_h, box_w = box['left'][-1, 0] - box['left'][0, 0],\
                   box['bottom'][-1, 1] - box['bottom'][0, 1]
    picture_transformed = transformation.prepare_picture(pic, box, to_resize=to_resize,
                                                         linear=linear)

    shift_inside = int((box_w - picture_transformed.shape[1]) / 2)
    shift = [
        min(box['left'][:, 0].min(), box['right'][:, 0].min()),
        box['left'][:, 1].min()  # + shift_inside
    ]

    img = _insert_pic(img, picture_transformed, shift, shift_inside, config,
                      img_index=img_index, img_source_mask=img_source_mask,
                      smooth_count=smooth_count)

    return img


def insert_right_pic(img, img_source_mask, pic, box, config, smooth_count=10):
    pic_h, pic_w, _ = pic.shape
    box_h, box_w = box['left'][-1, 0] - box['left'][0, 0], box['bottom'][-1, 1] - box['bottom'][0, 1]

    if box_w < box_h:
        pic = pic[:, :int(pic_w * (box_w / box_h))]

    return insert_pic(img, pic, box, config=config, to_resize=True, linear=True,
                      smooth_count=smooth_count, img_source_mask=img_source_mask)


def insert_pictures_into_frame(frame, frame_mask, pictures, boxes, config, smooth_count=10):
    img_work = frame.copy()

    if boxes['left'] is not None:
        img_work = insert_pic(img_work, pictures[0], config=config, box=boxes['left'],
                              linear=True, img_index=0, img_source_mask=frame_mask,
                              smooth_count=smooth_count)
    if boxes['right'] is not None:
        img_work = insert_right_pic(img_work, frame_mask, pictures[-1], config=config,
                                    box=boxes['right'], smooth_count=smooth_count)

    indices = list(filter(lambda i: config['dark'][i] > 0, range(1, 8)))
    images = list(map(lambda v: config['images'][v], indices))
    for i, pic, box in zip(indices, images, boxes['middle']):
        img_work = insert_pic(img_work, pic, box=box, config=config, linear=False,
                              img_index=i, img_source_mask=frame_mask,
                              smooth_count=smooth_count)

    # to cut the image, if right picture is not full
    if config['right-path']:
        img_work = img_work[:, :-20, :]

    return img_work


def cross_correlation(frame1, frame2, shift=10):
    f1, f2 = frame1, frame2
    res = []
    bix_in = slice(shift, -shift)
    for row in range(-shift, shift, 1):
        # for col in range(-shift, shift, 1):
        f_rolled = np.roll(f2, shift=row, axis=0)  # by row
        coef = np.mean(np.abs(f1[bix_in, bix_in].mean(axis=2)
                              - f_rolled[bix_in, bix_in].mean(axis=2)), axis=None)
        res.append([row, 0, coef])

    res = sorted(res, key=lambda v: v[-1])
    offset_row, offset_col, coef = res[0][0], res[0][1], res[0][-1]
    return (offset_row, offset_col), coef


def write_video(frames: list, path: str, sizes: Tuple[int, int] = None):
    if not frames:
        raise Exception('The frames are empty')

    print(f'Output shape: {frames[0].shape}')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h, w, _ = frames[0].shape
    if sizes:
        w, h = sizes

    out = cv2.VideoWriter(path, fourcc, 20.0, (w, h))

    for i in range(len(frames)):
        frame = frames[i]
        if sizes:
            frame = frame[:h, :w]

        out.write(frame)

    out.release()


def insert_pictures_into_video(frame_mask, pictures, config,
                               index: int, stop_frame=None):
    boxes = parsing.create_boxes(frame_mask, config=config)

    filename = f'video-{index}'
    vidcap = cv2.VideoCapture(f'data/{filename}.MOV')

    frames = []
    success, iteration = True, 0
    while success:
        if iteration % 10 == 0:
            print(f'Iteration {iteration}')
        success, frame = vidcap.read()
        if not success:
            break

        frame_out = insert_pictures_into_frame(frame, frame_mask, pictures, boxes,
                                               config=config, smooth_count=2)

        frames.append(frame_out)

        iteration += 1
        if stop_frame is not None and iteration > stop_frame:
            break

    write_video(frames, f'output/video/{filename}.mov')


def shift_frame(frame, offset_row):
    if (offset_row := round(offset_row)) != 0:
        frame = np.roll(frame, offset_row, axis=0)
        if offset_row > 0:
            frame[:offset_row] = (0, 0, 0)
        else:
            frame[offset_row:] = (0, 0, 0)

    return frame


def insert_pictures_into_video_shift(frame_mask, pictures, config,
                                          index: int, stop_frame=None, to_corr=False):
    boxes = parsing.create_boxes(frame_mask, config=config)

    filename = f'video-{index}'
    vidcap = cv2.VideoCapture(f'data/{filename}.MOV')

    frames, first_frame = [], None
    v_line = None
    success, iteration = True, 0
    while success:
        if iteration % 10 == 0:
            print(f'Iteration {iteration}')
        success, frame = vidcap.read()
        if not success:
            break

        frame_out = insert_pictures_into_frame(frame, frame_mask, pictures, boxes,
                                               config=config, smooth_count=2)

        frame_rolled1 = frame.copy() if to_corr else None
        frame_rolled2 = frame.copy() if to_corr else None

        if iteration == 0:
            first_frame = frame
            v_line = np.zeros((frame.shape[0], 2, 3), dtype='uint8')  # vertical line
        elif to_corr:
            rows, cols = slice(100, 400), slice(800, 1000)
            offset1, coef1 = cv2.phaseCorrelate(first_frame[rows, cols].mean(axis=2),
                                                frame[rows, cols].mean(axis=2))

            offset2, coef2 = cross_correlation(first_frame[rows, cols],
                                               frame[rows, cols], shift=10)

            print(offset1, round(coef2, 2), '|', offset2, coef2)
            frame_rolled1 = shift_frame(frame, offset_row=offset1[0])
            frame_rolled2 = shift_frame(frame, offset_row=offset2[0])

        if to_corr:
            frame_rolled_out1 = insert_pictures_into_frame(frame_rolled1, frame_mask, pictures,
                                                           boxes, config=config, smooth_count=2)

            frame_rolled_out2 = insert_pictures_into_frame(frame_rolled2, frame_mask, pictures,
                                                           boxes, config=config, smooth_count=2)

            w_size = frame.shape[1] // 3
            frame_out_merged = np.hstack((
                frame_out[:, :w_size - 1],
                v_line,
                frame_rolled_out1[:, w_size + 1: w_size * 2 - 1],
                v_line,
                frame_rolled_out2[:, w_size * 2 + 1:],
            ))
            frames.append(frame_out_merged)
        else:
            frames.append(frame)

        iteration += 1
        if stop_frame is not None and iteration > stop_frame:
            break

    write_video(frames, f'output/video_shifted/{filename}.mov', sizes=(1600, 400))


def plot_inserting_pictures(frame, frame_mask, pictures, boxes, config):
    img_out_1 = insert_pictures_into_frame(frame, frame_mask, pictures, boxes=boxes,
                                           config=config, smooth_count=0)

    fig0, ax0 = plt.subplots(1, 1, figsize=(5, 3))
    tools.imshow(img_out_1[210:380, 100:380], axes=[ax0], titles=['Проблема темных пикселей'],
                 fig=fig0, to_show=False)
    fig0.savefig('output/plots/inserting_border_problem.png')

    img_out_10 = insert_pictures_into_frame(frame, frame_mask, pictures, boxes=boxes,
                                            config=config, smooth_count=10)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    border_right, border_bottom = 700, 500
    tools.imshow(img_out_1[:border_bottom, :border_right],
                 img_out_10[:border_bottom, :border_right],
                 axes=list(axes),
                 titles=['Без сглаживания',
                         'Сглаживание с коэффициентом равным десяти'],
                 fig=fig)
    fig.savefig('output/plots/inserting.png')
    print(1)


def plot_inserting_final(frame, frame_mask, pictures, boxes, config, to_plot=False):
    img_out = insert_pictures_into_frame(frame, frame_mask, pictures, boxes=boxes,
                                         config=config, smooth_count=2)

    fig0, ax0 = plt.subplots(1, 1, figsize=(5, 3))
    if to_plot:
        tools.imshow(img_out[:400], axes=[ax0], titles=['Проблема темных пикселей'],
                     fig=fig0, to_show=True)
    fig0.savefig('output/plots/final_frame.png')
