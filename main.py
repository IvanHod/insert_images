import cv2

from src import transformation, parsing, inserting

images_config = {
    1: {
        'left': True,
        'right': True,
        'right-path': False,
        'dark': [.05, .001, .002, .02, .04, .08, .1, .12, .1],
        'images': [],
        'green_threshold': None,
        'green_filter': []
    },
    2: {
        'left': True,
        'right': True,
        'right-path': False,
        'dark': [.05, .001, .002, .02, .04, .08, .1, .12, .1],
        'images': [],
        'green_threshold': None,
        'green_filter': []
    },
    3: {
        'left': True,
        'right': False,
        'right-path': False,
        'dark': [.06, .01, .01, .04, .08, 0, 0, 0, 0],
        'images': [],
        'green_threshold': 160,
        'green_filter': [4]
    },
    4: {
        'left': False,
        'right': True,
        'right-path': True,
        'dark': [0, 0, .05, .1, .12, .18, .2, .22, .34],
        'images': [],
        'green_threshold': None
    },
    5: {
        'left': True,
        'right': True,
        'right-path': False,
        'dark': [.05, 0, .02, .02, .02, .01, .01, .01, .001],
        'images': [],
        'green_threshold': 160,
        'green_filter': [0]
    },
}


def read_data(index: int):
    img_source_mask = cv2.imread(f'data/mask/video-{index}.png')
    img_source = cv2.imread(f'data/zero-frames/video-{index}.png')

    pictures = []
    for i in range(1, 10):
        img = cv2.imread(f'data/{i}.jpg')
        pictures.append(img)
        for config_id, config in images_config.items():
            dark_coef = config['dark'][i - 1]
            if dark_coef > 0:
                img = (img.astype('float32') * (1 - dark_coef)).astype('uint8')

            config['images'].append(img)

    return img_source_mask, img_source, pictures


if __name__ == '__main__':
    img_index = 4
    img_source_mask, img_source, pictures = read_data(index=img_index)

    config = images_config[img_index]

    # Построение рамок
    boxes = parsing.plot_create_boxes(img_source, img_source_mask,
                                      config=config, to_plot=False)

    # Пример с отрисовкой перспективы
    # transformation.test_perspective_transform(img_source_mask, config,
    #                                           pictures, pic_index=0, to_plot=False)
    #
    # # Пример с отрисовкой афинного преобразования
    # transformation.test_affine_transform(img_source_mask, config,
    #                                      pictures, pic_index=1, to_plot=False)

    # Отрисовка конкретного вида библиотки (1-5), для демонстрации вставки изображений
    # inserting.plot_inserting_pictures(img_source, img_source_mask, pictures, boxes,
    #                                   config=config)

    # Отрисовка вставки последнего фрейма
    # inserting.plot_inserting_final(img_source, img_source_mask,
    #                                pictures, boxes, config=config, to_plot=True)

    inserting.insert_pictures_into_video(img_source_mask, pictures, config, index=img_index,
                                         stop_frame=None)
    print(1)
