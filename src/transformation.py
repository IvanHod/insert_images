import numpy as np
import cv2
from skimage import transform, util

from src import tools, parsing
from src.definitions import Point, Box, Polygon


def smooth_line(img_path, pic, line: np.ndarray, step: np.ndarray, coefs_save: list):
    h, w, _ = pic.shape

    line = line.copy()
    for coef in coefs_save:
        line_next = line[(line[:, 0] < h) & (line[:, 1] < w)]
        x, y = line_next[:, 0], line_next[:, 1]
        pic[x, y] = (pic[x, y] * (1 - coef)) + (img_path[x, y] * coef)

        line += step

    return pic.astype('uint8')


def to_resize_pic(pic, box):
    pic_h, pic_w, _ = pic.shape
    box_h, box_w = box['left'][-1, 0] - box['left'][0, 0], box['bottom'][-1, 1] - box['bottom'][0, 1]

    delta = max(pic_h / box_h, pic_w / box_w)
    pic_resized = cv2.resize(pic, (int(pic_w / delta), int(pic_h / delta)))

    return pic_resized


def matrix_for_perspective_transform(pic, box: dict, shift: tuple, delta=60, axes=None):
    """

    :param pic: Входное изображение
    :param box: Рамка для вставки
    :param shift: Смещение рамки относительно большого изображения
    :param delta:
    :param axes:
    :return:
    """
    h, w, _ = pic.shape
    width_max = box['top'][-1, 1] - box['top'][0, 1]
    shift_left = int((width_max - w) / 2)

    src = np.array([[0, 0], [0, w], [h, w], [h, 0]], dtype='float32')
    src += delta

    top, bottom = box['top'] - shift, box['bottom'] - shift
    dst = np.array([
        [top[shift_left, 0], top[shift_left:, 1].min()],
        [top[-shift_left, 0], top[:-shift_left, 1].max()],
        [bottom[-shift_left, 0], bottom[:-shift_left, 1].max()],
        [bottom[shift_left, 0], bottom[shift_left:, 1].min()],
    ], dtype='float32')
    img_width_new = dst[1, 1] - dst[0, 1]
    dst += delta

    pic_in = np.zeros((h + delta * 2, w + delta * 2, 3), dtype='uint8')
    pic_in[delta: h + delta, delta: w + delta] = pic

    M = cv2.getPerspectiveTransform(src[:, [1, 0]], dst[:, [1, 0]])
    if axes:
        tools.tables_show(src - delta, dst - delta, M, axes=axes,
                          titles=['Source matrix',
                                  'Destination matrix',
                                  'Matrix of a perspective transform']
                          )
    return M, pic_in, img_width_new


def affine_transform(pic, curve_top, curve_bottom, dsize):
    height, width, _ = pic.shape
    width_max = max(curve_top.shape[0], curve_bottom.shape[0])

    if curve_top.shape[0] < width_max:
        curve_top = np.vstack((curve_top, curve_top[-(width_max - curve_top.shape[0]):]))

    if curve_bottom.shape[0] < width_max:
        curve_bottom = np.vstack((curve_bottom, curve_bottom[-(width_max - curve_bottom.shape[0]):]))

    start_point = int((width_max - width) / 2)
    start_point = max(start_point, 0)  # todo
    end_point = start_point + width

    cols = np.repeat(np.arange(width), 2)

    src_rows = np.vstack((np.zeros(width), np.full(width, fill_value=height)))
    src = np.vstack((src_rows.T.flat, cols)).T  # col, row

    dst_cols = np.vstack((curve_top[start_point: end_point, 1], curve_bottom[start_point: end_point, 1]))
    dst_rows = np.vstack((curve_top[start_point: end_point, 0], curve_bottom[start_point: end_point, 0]))
    dst = np.vstack((dst_rows.T.flat, dst_cols.T.flat)).T

    img_width_new = curve_top[-1, 1] - curve_top[0, 1]

    pic_copy = np.zeros((height, width, 3), dtype='uint8')
    pic_copy[:] = pic

    tform = transform.PiecewiseAffineTransform()
    tform.estimate(src[:, [1, 0]], dst[:, [1, 0]])

    out_h, out_w, _ = pic.shape
    out = transform.warp(pic_copy, tform.inverse, clip=False, mode='constant', cval=0,
                         output_shape=dsize)

    return util.img_as_ubyte(out), img_width_new


def do_transform(pic: np.ndarray, box, kind: str, to_cut: bool = True,
                 offset_row=100, offset_col=200, table_axes=None):
    assert kind in {'linear', 'affine'}, 'kind must be "linear" or "affine"'

    shift = (box['left'][:, 0].min(), box['left'][:, 1].min())

    if kind == 'linear':
        M, pic_in, img_width_new = matrix_for_perspective_transform(pic, box=box, shift=shift,
                                                                    delta=min(offset_row, offset_col),
                                                                    axes=table_axes)
        warped = cv2.warpPerspective(pic_in, M, dsize=(pic_in.shape[1] + offset_row,
                                                       pic_in.shape[0] + offset_col))

    else:
        top, left = box['top'][:, 0].min(), box['left'][:, 1].min()
        pic_curve_top = box['top'] - np.array([top, left])
        pic_curve_bottom = box['bottom'] - np.array([top, left])

        warped, img_width_new = affine_transform(pic, pic_curve_top, pic_curve_bottom,
                                                 dsize=(pic.shape[1] + offset_row,
                                                        pic.shape[0] + offset_col))

    warped_cut = None
    if to_cut:
        # Cute picture without extra dark pixels
        warped_cut = warped[:, ~(warped == [0, 0, 0]).all(axis=2).all(axis=0)]
        warped_cut = warped_cut[~(warped_cut == [0, 0, 0]).all(axis=2).all(axis=1), :]

    return warped_cut, warped, img_width_new


def prepare_picture(pic, box, to_resize=True, linear=True):
    pic_resized = to_resize_pic(pic, box) if to_resize else pic

    transform_kind = 'linear' if linear else 'affine'
    warped_cut, warped, img_width_new = do_transform(pic_resized, box, kind=transform_kind)

    return warped_cut


def test_perspective_transform(img_source_mask, config, pictures,
                               pic_index, to_plot=False):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(16, 8))

    boxes = parsing.create_boxes(img_source_mask, config=config)
    box = boxes['left']  # left, right, middle[index]

    pic = to_resize_pic(pictures[pic_index], box)
    ax_picture = fig.add_subplot(2, 4, 1)  # rows, cols, index
    ax_box = fig.add_subplot(2, 4, 5)  # rows, cols, index

    # Without shifting
    _, warped0, _ = do_transform(pic, box=box, kind='linear', to_cut=False,
                                 offset_row=0, offset_col=0)
    ax_warped0 = fig.add_subplot(2, 4, 3)

    ax_table_scr = fig.add_subplot(3, 4, 2)
    ax_table_dst = fig.add_subplot(3, 4, 6)
    ax_table_M = fig.add_subplot(3, 4, 10)

    # Using shifting
    warped, warped1, _ = do_transform(pic, box=box, kind='linear', to_cut=True,
                                      offset_row=40, offset_col=40,
                                      table_axes=[ax_table_scr, ax_table_dst, ax_table_M])

    ax_warped1 = fig.add_subplot(2, 4, 7)

    # Cute picture without extra dark pixels
    ax_warped = fig.add_subplot(1, 4, 4)  # rows, cols, index

    tools.imshow(pic, tools.box_to_img(box), warped0, warped1, warped,
                 axes=[ax_picture, ax_box, ax_warped0, ax_warped1, ax_warped],
                 titles=['Мозаика', 'Место под вставку с перспективой',
                         'Обычная трансформация',
                         'Трансформация с "запасом"',
                         'Итог при перспективе'],
                 fig=fig, to_show=to_plot)
    fig.savefig('output/plots/perspective.png')


def test_perspective_transform_synthetic(picture, to_plot=False):
    from matplotlib import pyplot as plt

    dst = Box(
        Point(30, 30),
        Point(0, 400),
        Point(220, 370),
        Point(250, 0),
    )

    dsize = (400, 250)  # width, height
    pic = cv2.resize(picture, dsize)

    h_in, w_in = pic.shape[0], pic.shape[1]

    src = np.array([[0, 0], [0, w_in], [h_in, w_in], [h_in, 0]], dtype='float32')
    M = cv2.getPerspectiveTransform(src[:, [1, 0]], dst.points[:, [1, 0]])

    warped = cv2.warpPerspective(pic, M=M, dsize=dsize)

    fig = plt.figure(figsize=(16, 8))
    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

    ax_tl = fig.add_subplot(grid[0, 0])
    ax_bl = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[:, 1:])

    tools.imshow(pic, tools.box_obj_to_img(dst), warped,
                 axes=[ax_tl, ax_bl, ax_right],
                 titles=['Исходное изображение', 'Рамка под вставку', 'Выходное изображение'],
                 fig=fig,
                 to_show=to_plot)
    fig.savefig('output/plots/perspective_synthetic.png')


def test_affine_transform_synthetic(picture, to_plot=False):
    # shape of out img is (250, 400)
    from matplotlib import pyplot as plt

    dsize = (400, 250)  # width, height
    pic = cv2.resize(picture, dsize)

    curve_top_rows = 20 - np.round(np.arange(0, 400) ** 0.5)  # [0..20] by [0..400]
    func_cols = lambda rows: np.round((250 - rows) / 24.7)

    rows_vertical = np.arange(0, 250, 1)  # [0..250]

    fig = plt.figure(figsize=(16, 8))

    # generate box using exists functions, so translate it all lines and curves
    # to find intersections
    box = parsing.create_middle_boxes(lines_v=[
        np.vstack([rows_vertical, func_cols(rows_vertical)]).T,
        np.vstack([rows_vertical, func_cols(rows_vertical)]).T + np.array([0, 389])
    ], curves={
        'top': np.vstack([curve_top_rows, np.arange(0, 400)]).T,
        'bottom': np.vstack([curve_top_rows, np.arange(0, 400)]).T + np.array([229, 0])
    })[0]

    box = Polygon(box['left'], box['top'], box['right'], box['bottom'])

    src = np.vstack((
        np.vstack((np.zeros(box.top.shape[0]), box.top[:, 1])).T,  # top
        np.vstack((np.ones(box.bottom.shape[0]) * box.height, box.bottom[:, 1])).T,  # bottom
    )).astype('int')
    dst = np.vstack((box.top, box.bottom)).astype('int')

    tform = transform.PiecewiseAffineTransform()
    tform.estimate(src[:, [1, 0]], dst[:, [1, 0]])

    out_h, out_w, _ = pic.shape
    warped = transform.warp(pic, tform.inverse, clip=False, order=1,
                            mode='constant', cval=0,
                            output_shape=(box.height, box.width))
    warped = (warped * 255).astype('uint8')

    fig = plt.figure(figsize=(16, 8))
    grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

    ax_tl = fig.add_subplot(grid[0, 0])
    ax_bl = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[:, 1:])

    tools.imshow(pic, tools.box_to_img(box), warped,
                 axes=[ax_tl, ax_bl, ax_right],
                 titles=['Исходное изображение', 'Рамка под вставку', 'Выходное изображение'],
                 fig=fig,
                 to_show=to_plot)

    fig.savefig('output/plots/affine_synthetic.png')


def test_affine_transform(img_source_mask, config, pictures, pic_index,
                          to_plot=False):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(16, 8))

    boxes = parsing.create_boxes(img_source_mask, config=config)
    box = boxes['middle'][0]  # left, right, middle[index]

    pic = to_resize_pic(pictures[pic_index], box)

    ax_picture = fig.add_subplot(2, 3, 1)  # rows, cols, index
    ax_box = fig.add_subplot(2, 3, 4)  # rows, cols, index

    ax_warped0 = fig.add_subplot(2, 3, 2)
    _, warped0, _ = do_transform(pic, box=box, kind='affine', to_cut=False,
                                 offset_row=0, offset_col=0)

    warped_cut, warped, _ = do_transform(pic, box, kind='affine', to_cut=True,
                                         offset_row=0, offset_col=80)

    ax_warped = fig.add_subplot(2, 3, 5)  # row 2, col 4
    ax_warped_cut = fig.add_subplot(1, 3, 3)  # row 1, col 4

    tools.imshow(pic, tools.box_to_img(box), warped0, warped, warped_cut,
                 axes=[ax_picture, ax_box, ax_warped0, ax_warped, ax_warped_cut],
                 titles=['Мозаика', 'Деформированное место под вставку',
                         'Обычная трансформация',
                         'Трансформация с "запасом"',
                         'Итог при аффинном преобразовании'],
                 fig=fig, to_show=to_plot)
    fig.savefig('output/plots/affine.png')
