# --- Среднее время работы программы для 1 кадра равно 0.03577457806643318 сек.
# --- Протестировано на MacBook Air (Retina, 13-inch, 2019) [1.6 GHz 2‑ядерный процессор IntelCorei5] ---


import time
import warnings

import cv2
import mediapipe as mp
import numpy as np

from Face import Face

warnings.filterwarnings("ignore")
import logging

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)


def crop_face_dist(left: list, right: list):
    left.sort(key=lambda elem: elem[1])
    right.sort(key=lambda elem: elem[1])


def main():
    with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:
        iteration = 0
        while cap.isOpened():

            start_time = time.time()

            success, image = cap.read()

            # --- Размеры исходного окна ---
            h, w, c = image.shape

            if not success:
                logging.warning("Camera Trouble")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # --- Собираем "особенные" точки лица ---
                    keypoints = []
                    x_points = []
                    y_points = []

                    for data_point in face_landmarks.landmark:
                        keypoints.append((data_point.x, data_point.y))
                        x_points.append(int(data_point.x * w))
                        y_points.append(int(data_point.y * h))

                x_points = np.array(x_points)
                y_points = np.array(y_points)

                # Центр прямоугольника, который образуется границами лица. Вокруг него будет происходить вращение.
                x0, y0 = (min(x_points) + max(x_points)) / 2, (min(y_points) + max(y_points)) / 2

                # Объект класса Face.
                face = Face(keypoints, x_points, y_points, x0, y0, w, h)

                # Находим угол поворота лица в радианах.
                angle = face.angleFind()

                # Находим угол поворота лица в градусах.
                angle_deg = face.angleFind_deg()

                extreme_points, dist_x, dist_y = face.rotateFace(angle, image)[0], \
                                                 int(face.rotateFace(angle, image)[1]), \
                                                 int(face.rotateFace(angle, image)[2])

                # Границы лица (простое нахождение максимумов и минимумов по точкам)
                x_max = max(x_points)
                y_max = max(y_points)
                x_min = min(x_points)
                y_min = min(y_points)

                # Углы лица
                up_Left_x, up_Left_y = extreme_points['up_Left'][0], extreme_points['up_Left'][1]
                up_Right_x, up_Right_y = extreme_points['up_Right'][0], extreme_points['up_Right'][1]
                down_Left_x, down_Left_y = extreme_points['down_Left'][0], extreme_points['down_Left'][1]
                down_Right_x, down_Right_y = extreme_points['down_Right'][0], extreme_points['down_Right'][1]

                # Отсортированные верхние и нижние точки лица
                up_points, down_points = zip(face.getSortPoints(extreme_points))

                # necessary - точки которые находятся на "необходимой" оси прямоугольника (параллельно оси абсцисс)
                # unnecessary - точки которые не находятся на "необходимой" оси прямоугольника (параллельно оси ординат)
                necessary_up = up_points[0][0]
                unnecessary_up = up_points[0][1]

                necessary_down = down_points[0][0]
                unnecessary_down = down_points[0][1]

                left_up_corner = (unnecessary_down[0], necessary_up[1])
                right_up_corner = (unnecessary_up[0], unnecessary_up[1])

                left_down_corner = (unnecessary_down[0], unnecessary_down[1])
                right_down_corner = (unnecessary_up[0], necessary_down[1])

                prev_frame = image[left_up_corner[1]:right_down_corner[1],
                             min(left_up_corner[0], right_down_corner[0]):max(left_up_corner[0], right_down_corner[0])]

                prev_width = abs(
                    min(left_up_corner[0], right_down_corner[0]) - max(left_up_corner[0], right_down_corner[0])) + 1
                prev_height = abs(left_up_corner[1] - right_down_corner[1]) + 1

                rectangle_x_points = np.arange(min(left_up_corner[0], right_up_corner[0]),
                                               max(left_up_corner[0], right_up_corner[0]) + 1)
                rectangle_y_points = np.arange(min(left_up_corner[1], left_down_corner[1]),
                                               max(left_up_corner[1], left_down_corner[1]) + 1)

                # Поворот окна матрицей поворота
                M = cv2.getRotationMatrix2D((prev_width / 2, prev_height / 2), angle_deg, 1.0)

                # Афинное преобразование повернутого окна
                rotated = cv2.warpAffine(prev_frame, M, (prev_width, prev_height))

                h_rotated, w_rotated, _ = map(int, rotated.shape)
                center_h = h_rotated // 2
                center_w = w_rotated // 2

                rotated = rotated[center_h - dist_x // 2: center_h + dist_x // 2,
                          center_w - dist_y // 2: center_w + dist_y // 2]

                # Построение нормализованного прямоугольника
                cv2.rectangle(image, left_up_corner, right_down_corner, (255, 0, 255), 2)

                print(f'Время обработки кадра : {(time.time() - start_time)}')
                iteration += 1

                if results.multi_face_landmarks:
                    cv2.imshow('Face', prev_frame)
                    cv2.imshow('FaceRotate', rotated)
                    cv2.imshow('Result', image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
    cap.release()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.warning('Something Went Wrong...')
