import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cv2 import cv2
from sklearn.linear_model import LinearRegression

from src import tools

threshold = 100


def interp(line, min_p, max_p, border=None):
    X = line[:, 0].reshape((line.shape[0], 1))
    m = LinearRegression().fit(X, line[:, 1])
    if border is None or abs(m.coef_[0]) < border:  # 0.2 по умолчанию для вертикальных линий
        rows = np.arange(min_p, max_p)
        cols = m.predict(rows.reshape((rows.shape[0], 1)))

        return np.vstack((rows, cols)).T.astype('int32')

    return None


def approximate_lines(lines):
    lines_out = []
    for line in lines:
        if len(line) < 5:
            continue

        line = np.array(line)
        line = interp(line, min_p=1, max_p=line[:, 0].max(), border=0.2)

        if line is not None:
            lines_out.append(line)

    return lines_out


def intersect2d(l1, l2, name=None):
    l1_set = set(map(tuple, l1))
    l2_set = set(map(tuple, l2))

    intersect = l1_set & l2_set
    if len(intersect) > 1:
        intersect = sorted(intersect, key=lambda v: v[1])[-1:]

    if len(intersect) == 1:
        p = list(intersect)[0]

        l1_index = np.where((l1[:, 0] == p[0]) & (l1[:, 1] == p[1]))[0]
        l2_index = np.where((l2[:, 0] == p[0]) & (l2[:, 1] == p[1]))[0]

        return p, l1_index[0], l2_index[0]
    else:
        raise Exception(f'The intersect is not single: {len(intersect)} ({intersect})')


def create_left_box(lines_v, lines_h):
    print('create_left_box')
    line_vl, line_vr = lines_v[0], lines_v[1]
    left, right = line_vl[:, 1].min(), line_vr[:, 1].max()

    line_top, line_bottom = tuple(sorted(lines_h, key=lambda l: l[0, 1]))  # sort by rows

    # intrerpolation works using 0 axis like X, 1 axis like y
    # to interpolate horizontally, it is need to inverse [:, [1, 0]]
    line_top = interp(line_top[:, [1, 0]], min_p=left, max_p=right + 1)[:, [1, 0]]
    line_bottom = interp(line_bottom[:, [1, 0]], min_p=left, max_p=right + 1)[:, [1, 0]]

    return create_middle_boxes(lines_v, {'top': line_top, 'bottom': line_bottom})[0]


def create_right_box(lines_v, lines_h):
    if len(lines_v) == 0:
        return []

    if len(lines_v) == 1:
        rows = np.arange(lines_h[0][-1, 0], lines_h[1][-1, 0] + 1)
        cols = np.array([max(lines_h[0][:, 1].max(), lines_h[1][:, 1].max())] * rows.size)

        lines_v.append(np.vstack((rows, cols)).T)

    return create_left_box(lines_v, lines_h)


def create_middle_boxes(lines_v, curves):
    print(f'Find middle boxes using {len(lines_v)} lines')
    boxes = []
    for i in range(0, len(lines_v), 2):
        print(f'Find middle box: {i}, lines: {i}, {i + 1}')
        l1, l2 = lines_v[i], lines_v[i + 1]

        p, l_lt_index, l_tl_index = intersect2d(l1, curves['top'], name='left-top')
        p, l_rt_index, l_tr_index = intersect2d(l2, curves['top'], name='right-top')

        p, l_lb_index, l_bl_index = intersect2d(l1, curves['bottom'], name='left-bottom')
        p, l_rb_index, l_br_index = intersect2d(l2, curves['bottom'], name='right-bottom')

        box = {
            'top': curves['top'][l_tl_index: l_tr_index + 1],  # top
            'right': l2[l_rt_index: l_rb_index + 1],  # right
            'bottom': curves['bottom'][l_bl_index: l_br_index + 1],  # bottom
            'left': l1[l_lt_index: l_lb_index + 1],  # left
        }

        boxes.append(box)

    return boxes


def create_boxes(img_source_mask, config: dict):
    lines_v, lines_h, curves = find_lines_cv2(img_source_mask)

    left_box = create_left_box(lines_v[:2], lines_h) if config['left'] else None

    middle_index = 2 if config['left'] else 0
    middle_index_end = len(lines_v) - (config['right'] * 2 - config['right-path'])  # 7 центральных рамок
    print(middle_index, middle_index_end, len(lines_v))
    middle_boxes = create_middle_boxes(lines_v[middle_index: middle_index_end], curves)

    right_index = 1 if config['right-path'] else 2
    right_box = create_right_box(lines_v[-right_index:], lines_h) if config['right'] else None

    return {'left': left_box, 'middle': middle_boxes, 'right': right_box}


def contours_to_lines(c, axis):
    """

    :param c:
    :param axis: row or column
    :return:
    """
    c = pd.DataFrame(c, columns=['col', 'row'])
    c = c.groupby(axis, as_index=False).mean().sort_values(axis)
    return c.astype(int).values


def find_lines_cv2(img_mask):
    """

    :param img_mask:
    :return: lines_v - rows are grow: [[1, X], [2, X], ...]
    """
    blue_mask = ((img_mask[:, :, 0] > 240) & (img_mask[:, :, 1] < 20)).astype('uint8') * 255
    green_mask = ((img_mask[:, :, 1] > 240) & (img_mask[:, :, 0] < 20)).astype('uint8') * 255

    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    v_lines, h_lines, curves = [], [], []
    for contour in contours:
        c = contour[:, 0, :]
        if c[:, 1].max() - c[:, 1].min() > c[:, 0].max() - c[:, 0].min():
            # it is vertical, height more than width
            v_lines.append(contours_to_lines(c, axis='row'))
        else:
            # group by col, to mean the rows
            curves.append(contours_to_lines(c, axis='col')[:, [1, 0]])

    for c in contours_blue:
        # it is need to reshape lines, set rows to first place and cols to second
        h_lines.append(contours_to_lines(c[:, 0, :], axis='col')[:, [1, 0]])

    # plt.imshow(blue_mask)
    # plt.show()

    curves = sorted(curves, key=lambda c: c[0, 0])  # sort by row
    curves = {'top': curves[0], 'bottom': curves[1]}

    v_lines = sorted(v_lines, key=lambda l: l[0, 1])  # sort by column, left to right
    v_lines = approximate_lines(v_lines)
    return v_lines, h_lines, curves


def plot_create_boxes(img, img_source_mask, config, to_plot=False):
    green_mask = ((img_source_mask[:, :, 1] > 240) & (img_source_mask[:, :, 0] < 20)).astype('uint8') * 255
    max_row = np.where(green_mask.sum(axis=1) > 0)[0][-1]

    boxes = create_boxes(img_source_mask, config=config)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(img_source_mask)

    img_out = img.copy()
    if left := boxes['left']:
        lines = list(map(lambda v: v[:, [1, 0]], left.values()))
        cv2.polylines(img_out, lines, False, (0, 0, 0), thickness=3)

    for box in boxes['middle']:
        lines = list(map(lambda v: v[:, [1, 0]], box.values()))
        cv2.polylines(img_out, lines, False, (0, 0, 0), thickness=3)

    tools.imshow(img_source_mask[: max_row, : 1000],
                 img_out[: max_row, : 1000],
                 axes=[axes[0], axes[1]],
                 titles=['Размеченное изображение', 'Извлеченные рамки'],
                 fig=fig, to_show=to_plot)
    fig.savefig('output/plots/boxes.png')

    return boxes
