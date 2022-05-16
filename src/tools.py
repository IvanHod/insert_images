import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


def imshow(*args, axes=None, titles=None, fig=None, to_show=False):
    img_len = len(args)
    if axes is None:
        fig, axes = plt.subplots(1, img_len, figsize=(16, 8))
    elif not isinstance(axes, list):
        axes = [axes]

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    kwargs = {}
    for i, (ax, img) in enumerate(zip(axes, args)):
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            kwargs.update({'cmap': 'gray', 'vmin': 0., 'vmax': 1.})

        ax.imshow(img, **kwargs)

        if title := (titles[i] if titles else None):
            ax.set_title(title)
        ax.set_axis_off()

    if fig is not None:
        fig.tight_layout()

    if to_show:
        fig.show()


def tables_show(*args, axes=None, titles=None):
    for i, (ax, table) in enumerate(zip(axes, args)):
        table = table.round(2).tolist()
        ax.table(table, loc='center')
        if title := (titles[i] if titles else None):
            ax.set_title(title, y=0.7)
        ax.set_axis_off()


def box_to_img(box):
    left, top = box['left'][:, 1], box['top'][:, 0]
    right, bottom = box['right'][:, 1], box['bottom'][:, 0]

    left_full = np.hstack((np.repeat(0, max(box['left'][0, 0] - top.min() - 1, 0)), left,
                           np.repeat(0, max(bottom.max() - box['left'][-1, 0] - 1, 0))))

    right_full = np.hstack((np.repeat(right.max(), max(box['right'][0, 0] - top.min() - 1, 0)), right,
                            np.repeat(right.max(), max(bottom.max() - box['right'][-1, 0] - 1, 0))))

    top_full = np.hstack((np.repeat(0, max(box['top'][0, 1] - left.min(), 0)),
                          top,
                          np.repeat(0, max(right.max() - box['top'][-1, 1], 0))))

    bottom_full = np.hstack((np.repeat(bottom.max(), max(box['bottom'][0, 1] - left.min(), 0)),
                             bottom,
                             np.repeat(bottom.max(), max(right.max() - box['bottom'][-1, 1], 0))))

    assert top_full.size == bottom_full.size, 'Must be equal'
    assert left_full.size == right_full.size, 'Must be equal'

    # x - rows, y - cols
    x, y = np.arange(left_full.size) + top.min(),\
           np.arange(top_full.size) + left.min()
    xx, yy = np.meshgrid(y, x)

    indices = (yy >= top_full) & (yy <= bottom_full) &\
              ((xx.T >= left_full) & (xx.T < right_full)).T

    img = (~indices).astype(float)

    return img
