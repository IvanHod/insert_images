import numpy as np
from cv2 import cv2
from matplotlib import pyplot as plt


def imshow(*args, axes=None, titles=None, fig=None):
    img_len = len(args)
    if axes is None:
        fig, axes = plt.subplots(1, img_len, figsize=(16, 8))
    elif not isinstance(axes, list):
        axes = [axes]

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for i, (ax, img) in enumerate(zip(axes, args)):
        if len(args[1].shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        ax.imshow(img)
        if title := (titles[i] if titles else None):
            ax.set_title(title)
        ax.set_axis_off()

    if fig is not None:
        fig.tight_layout()
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

    top_min, bottom_max = top.min(), bottom.max()
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

    # x - rows, y - cols
    x, y = np.arange(bottom.max() - top.min()) + top.min(),\
           np.arange(max(top.size, bottom.size)) + left.min()
    xx, yy = np.meshgrid(y, x)

    indices = (yy >= top_full) & (yy <= bottom_full) &\
              ((xx.T >= left_full) & (xx.T < right_full)).T

    img = (~indices).astype(float)

    return img