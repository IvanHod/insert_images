import numpy as np
from sklearn.linear_model import LinearRegression

threshold = 100


def interp(line, min_p, max_p, border=None):
    m = LinearRegression().fit(line[:, 0].reshape((line.shape[0], 1)), line[:, 1])
    if border is None or abs(m.coef_[0]) < border:  # 0.2 по умолчанию для вертикальных линий
        rows = np.arange(min_p, max_p)
        cols = m.predict(rows.reshape((rows.shape[0], 1)))

        return np.vstack((rows, cols)).T.astype('int32')

    return None


def approximate_lines(lines):
    lines_out = []
    for line_info in lines:
        if len(line_info['line']) < 5:
            continue

        line = np.array(line_info['line'])
        line = interp(line, min_p=1, max_p=line[:, 0].max(), border=0.2)

        if line is not None:
            lines_out.append(line)

    return lines_out


def add_points_to_lines(lines, points: list, row, shift=1):
    for p in points:
        lines_filtered = list(filter(lambda l: l['border'][0] <= p <= l['border'][1], lines))

        if not lines_filtered:
            lines.append({'border': (p - shift, p + shift), 'line': [(row, p)]})
        else:
            if len(lines_filtered) > 1:  # дошли до дуги
                lines_filtered = sorted(lines_filtered, key=lambda l: -len(l['line']))

            line_info = lines_filtered[0]
            if line_info['line'][-1][0] - row < 2:
                line_info['line'].append((row, p))

                border = line_info['border']
                line_info['border'] = (min(p - shift, border[0]), max(p + shift, border[1]))  # обновить границы

    return lines


def lines_get_splits(array, threshold=100, i=None):
    indices = np.where(array > threshold)[0]

    if indices.shape[0] == 1:
        return indices

    elif indices.shape[0] > 1:
        diff, splits = np.diff(indices), []

        if (diff > 5).any():
            split_indices = np.where(diff > 5)[0] + 1
            splits = np.split(indices, split_indices)
        else:
            splits = [indices]

        splits = list(map(lambda v: int(v.mean()), splits))

        return splits

    return None


def find_vertical_lines(img_source_mask):
    green_mask = (img_source_mask[:, :, 1] > 240) & (img_source_mask[:, :, 0] < 10)
    green_mask = green_mask.astype('uint8') * 255

    lines = []  # (border [left, right], line)

    for i, row in enumerate(range(green_mask.shape[0] - 1, 0, -1)):
        img_row = green_mask[row, :]

        splits = lines_get_splits(img_row, threshold=threshold)
        if splits is not None:
            lines = add_points_to_lines(lines, splits, row)

    lines = approximate_lines(lines)  # list(np.array([row, col]))
    return sorted(lines, key=lambda l: l[0][1])


def add_h_points_to_lines(lines, points: list, col, shift=1):
    """
    """
    for row in points:
        lines_filtered = list(filter(lambda l: l[-1][0] - shift <= row <= l[-1][0] + shift, lines))

        if not lines_filtered:
            lines.append([(row, col)])
        else:
            if len(lines_filtered) > 1:  # Линии на одних и тех же строках
                # Фильтруем, чтобы первая запись была правее всего
                lines_filtered = sorted(lines_filtered, key=lambda l: -l[-1][1])

            line = lines_filtered[0]

            #             if col == 1373:
            #                 print(lines_filtered, col - line[-1][1])
            #                 return None
            if col - line[-1][1] < 2:
                line.append((row, col))
            else:
                lines.append([(row, col)])

    return lines


def find_horizontal_lines(img_source_mask, config: dict, color=(0, 0, 255)):
    blue_mask = (img_source_mask[:, :, 0] > 240) & (img_source_mask[:, :, 1] < 30)  # blue lines
    blue_mask = blue_mask.astype('uint8') * 255
    #     imshow(blue_mask)

    lines = []  # (border [left, right], line)

    for i, col in enumerate(range(0, blue_mask.shape[1])):
        img_col = blue_mask[:, col]

        splits = lines_get_splits(img_col, threshold=threshold, i=col)  # returns rows

        if splits is not None:
            lines = add_h_points_to_lines(lines, splits, col=col)
            if lines is None:
                break

    assert len(lines) in {2,
                          4}, f'Найдено неправильное кол-во {len(lines)} горизонтальных линий, с цветом маски: "blue"'
    res = {'left': None, 'right': None}
    for side in ['left', 'right']:
        i = 0 if side == 'left' or not config['left'] else 2
        if config[side]:
            index_top = i + (0 if lines[i][0][0] < lines[i + 1][0][0] else 1)
            res[side] = (np.array(lines[index_top]), np.array(lines[1 - index_top]))

    return res


def add_curve_points_to_curves(curves, points: list, col, shift=1):
    #     if col == 489: print(points, col)
    for row in points:
        curves_filtered = list(filter(lambda l: l[-1][0] - 2 <= row <= l[-1][0] + 2, curves))

        if not curves_filtered:
            curves.append([(row, col)])
        else:
            if len(curves_filtered) > 1:  # Линии на одних и тех же строках
                raise Exception('...')

            curve = curves_filtered[0]
            if curve[-1][1] - col < 2:
                curve.append((row, col))

    curves_out = []
    for c in curves:
        right_point = c[-1]  # столбец последней (наиправейшей) точки
        if col - right_point[1] < 5 or len(c) > 5:
            curves_out.append(c)
    #         elif col > 570:
    #             print('remove_curve', col, c)

    return curves_out


def find_horizontal_curves(img_source_mask, color='green'):
    green_mask = (img_source_mask[:, :, 1] > 240) & (img_source_mask[:, :, 0] < 20)
    green_mask = green_mask.astype('uint8') * 255

    curves = []  # (border [left, right], line)

    for i, col in enumerate(range(0, green_mask.shape[1])):
        img_col = green_mask[:, col]

        splits = lines_get_splits(img_col, threshold=threshold, i=col)  # returns rows
        #         if col == 489: print('splits:', splits)
        if splits is not None:
            curves = add_curve_points_to_curves(curves, splits, col=col)

    curves = add_curve_points_to_curves(curves, points=[], col=green_mask.shape[1] - 1)

    assert len(
        curves) == 2, f'Найдено неправильное кол-во {len(curves)} горизонтальных кривых, с цветом маски: "{color}"'
    index_top = 0 if curves[0][0][0] < curves[1][0][0] else 1
    return {'top': np.array(curves[index_top]), 'bottom': np.array(curves[1 - index_top])}


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

    line_top = interp(lines_h[0][:, [1, 0]], min_p=left, max_p=right + 1)[:, [1, 0]]
    line_bottom = interp(lines_h[1][:, [1, 0]], min_p=left, max_p=right + 1)[:, [1, 0]]

    return create_middle_boxes(lines_v, {'top': line_top, 'bottom': line_bottom})[0]


def create_right_box(lines_v, lines_h):
    if len(lines_v) == 0:
        return []

    if len(lines_v) == 1:
        rows = np.arange(lines_h[0][-1, 0], lines_h[1][-1, 0] + 1)
        cols = np.array([max(lines_h[0][:, 1].max(), lines_h[1][:, 1].max())] * rows.size)

        lines_v.append(np.vstack((rows, cols)).T)

    return create_left_box(lines_v, lines_h)

    line_left, line_right = lines_v[0], None if len(lines_v) == 1 else lines_v[0]


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
    curves = find_horizontal_curves(img_source_mask)
    lines_h = find_horizontal_lines(img_source_mask, config)
    lines_v = find_vertical_lines(img_source_mask)

    left_box = create_left_box(lines_v[:2], lines_h['left']) if config['left'] else None

    middle_index = 2 if config['left'] else 0
    middle_index_end = len(lines_v) - (config['right'] * 2 - config['right-path'])  # 7 центральных рамок
    print(middle_index, middle_index_end, len(lines_v))
    middle_boxes = create_middle_boxes(lines_v[middle_index: middle_index_end], curves)

    right_index = 1 if config['right-path'] else 2
    right_box = create_right_box(lines_v[-right_index:], lines_h['right']) if config['right'] else None

    return {'left': left_box, 'middle': middle_boxes, 'right': right_box}


# boxes = create_boxes(img_source_mask, config=images_config[index])
# boxes


# curves = find_horisontal_curces(img_source_mask)


# lines_h = find_horisontal_lines(img_source_mask, config=images_config[index])
# f'left found: {lines_h["left"] is not None}, right found: {lines_h["right"] is not None}'


# lines_v = find_vertical_lines(img_source_mask)
# len(lines_v)