from dataclasses import dataclass

from PIL import Image
import cv2

from src.definitions import Border


@dataclass
class Sizes:
    width: int
    height: int


def video_to_gif(video_path: str, output_name: str, step: int, border: Border = None):
    output_name_full = f'output/gif/{output_name}'

    vidcap = cv2.VideoCapture(video_path)

    frames = []
    success, iteration = True, 0
    while success:
        if iteration % 10 == 0:
            print(f'Iteration {iteration}')

        success, frame = vidcap.read()
        if not success:
            break

        if iteration % step == 0:
            frame_cut = frame
            if border:
                frame_cut = frame[border.top: border.bottom, border.left: border.right]
            frames.append(Image.fromarray(cv2.cvtColor(frame_cut, cv2.COLOR_BGR2RGB)))

        iteration += 1

    if frames:
        img_first = frames[0]

        duration = 40  # 5 seconds
        loop_forever = 0
        img_first.save(fp=output_name_full.replace('.gif', '.png'), format='PNG')

        img_first.save(fp=output_name_full, format='GIF', append_images=frames[1:],
                       save_all=True, duration=duration, optimize=False,
                       loop=loop_forever)


if __name__ == '__main__':
    # Преобразовать и обрезать в gif видео
    video_path_ = 'output/video/video-4.mov'
    video_to_gif(video_path_, 'video-4.gif', step=10,
                 border=Border(right=1600, bottom=400))

    # Преобразовать в gif видео со стабилизацией
    video_path_ = 'output/video_shifted/video-4.mov'
    video_to_gif(video_path_, 'video-4-merged.gif', step=10, border=None)

    print(1)
