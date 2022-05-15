import numpy as np


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


def find_vertical_lines(img_source_mask, threshold=100):
    from src import parsing

    green_mask = (img_source_mask[:, :, 1] > 240) & (img_source_mask[:, :, 0] < 10)
    green_mask = green_mask.astype('uint8') * 255

    lines = []  # (border: [left, right], line: [[], ...])

    for i, row in enumerate(range(green_mask.shape[0] - 1, 0, -1)):
        img_row = green_mask[row, :]

        splits = lines_get_splits(img_row, threshold=threshold)
        if splits is not None:
            lines = add_points_to_lines(lines, splits, row)

    lines = list(map(lambda l: l['line'], lines))
    lines = parsing.approximate_lines(lines)  # list(np.array([row, col]))
    return sorted(lines, key=lambda l: l[0][1])


def find_horizontal_curves(img_source_mask, color='green', threshold=100):
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

    assert len(curves) == 2,\
        f'Найдено неправильное кол-во {len(curves)} горизонтальных кривых, с цветом маски: "{color}"'

    index_top = 0 if curves[0][0][0] < curves[1][0][0] else 1
    return {'top': np.array(curves[index_top]), 'bottom': np.array(curves[1 - index_top])}


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


def find_horizontal_lines(img_source_mask, config: dict, threshold=100, color=(0, 0, 255)):
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
