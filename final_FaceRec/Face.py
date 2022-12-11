import math

import cv2
import pandas as pd


class Face:
    def __init__(self, keypoints, x_points, y_points, x0, y0, w, h):
        self.keypoints = keypoints
        self.x_points = x_points
        self.y_points = y_points
        self.x0 = x0
        self.y0 = y0
        self.w = w
        self.h = h
        self.extremums = None

    def angleFind(self) -> float:
        """
        Нахождение угла поворота лица.
        Находим уравнение прямой, которую провели между глазами (id: 159, 386).
        Находя arctg (angle), возвращаем угол поворота лица.
        :return: Угол Поворота в Радианах
        """

        def create_line(x1: int, y1: int, x2: int, y2: int) -> list:
            """Two-Point Straight Line Function

            Keyword arguments:
                x1 -- First point on the x-axis.
                y1 -- First point on the y-axis.
                x2 -- 2nd point on the x-axis.
                y2 -- 2nd point on the y-axis.
            """

            k = (y1 - y2) / (x1 - x2)
            b = y2 - k * x2
            return [k, b]

        matrix = self.keypoints
        w = self.w
        h = self.h

        df = pd.DataFrame(matrix, columns=['x', 'y'])

        # --- Right eye index. ---

        x1, y1 = df.iloc[159]['x'] * w, df.iloc[159]['y'] * h

        # --- Left eye index. ---

        x2, y2 = df.iloc[386]['x'] * w, df.iloc[386]['y'] * h

        k, b = create_line(x1, y1, x2, y2)[0], create_line(x1, y1, x2, y2)[1]

        # --- Calculating the angle of the face. ---

        angle_r = math.atan(k)  # [Rad]
        angle_d = math.degrees(angle_r)  # [Degrees]

        return round(angle_r, 20)

    def angleFind_deg(self) -> float:
        """
        Нахождение угла поворота лица.
        Находим уравнение прямой, которую провели между глазами (id: 159, 386).
        Находя arctg (angle), возвращаем угол поворота лица.
        :return: Угол Поворота в Радианах
        """

        def create_line(x1: int, y1: int, x2: int, y2: int) -> list:
            """
            Нахождение уравнения прямой по двум точкам.
            Keyword arguments:
                x1 -- First point on the x-axis.
                y1 -- First point on the y-axis.
                x2 -- 2nd point on the x-axis.
                y2 -- 2nd point on the y-axis.
            """

            k = (y1 - y2) / (x1 - x2)
            b = y2 - k * x2
            return [k, b]

        matrix = self.keypoints
        w = self.w
        h = self.h

        df = pd.DataFrame(matrix, columns=['x', 'y'])

        # --- Right eye index. ---

        x1, y1 = df.iloc[159]['x'] * w, df.iloc[159]['y'] * h

        # --- Left eye index. ---

        x2, y2 = df.iloc[386]['x'] * w, df.iloc[386]['y'] * h

        k, b = create_line(x1, y1, x2, y2)[0], create_line(x1, y1, x2, y2)[1]

        # --- Calculating the angle of the face. ---

        angle_r = math.atan(k)  # [Rad]
        angle_d = math.degrees(angle_r)  # [Degrees]

        return round(angle_d, 20)

    def get_extreme_points(self, angle):
        """Функция, поворачивающая точки на заданный угол (Используем матрицу поворота).
        Keyword arguments:
            angle -- Landmark list.
            x -- Points of the face along the abscissa.
            y -- Points of the face along the ordinates.
            x0 -- The point around which the turn occurs.
            y0 -- The point around which the turn occurs.
        """
        x = self.x_points
        y = self.y_points
        x0 = self.x0
        y0 = self.y0

        # Left Top Point of Rectangle.
        up_left_x1 = int(-1 * math.sin(angle) * (min(y) - y0) + math.cos(angle) * (min(x) - x0) + x0)
        up_left_y1 = int(math.cos(angle) * (min(y) - y0) + math.sin(angle) * (min(x) - x0) + y0)

        # Right Upper point of the rectangle.
        up_right_x1 = int(-1 * math.sin(angle) * (min(y) - y0) + math.cos(angle) * (max(x) - x0) + x0)
        up_right_y1 = int(math.cos(angle) * (min(y) - y0) + math.sin(angle) * (max(x) - x0) + y0)

        # Right Lower Point of Rectangle.
        down_right_x1 = int(-1 * math.sin(angle) * (max(y) - y0) + math.cos(angle) * (max(x) - x0) + x0)
        down_right_y1 = int(math.cos(angle) * (max(y) - y0) + math.sin(angle) * (max(x) - x0) + y0)

        # Left Lower Point of the Rectangle.
        down_left_x1 = int(-1 * math.sin(angle) * (max(y) - y0) + math.cos(angle) * (min(x) - x0) + x0)
        down_left_y1 = int(math.cos(angle) * (max(y) - y0) + math.sin(angle) * (min(x) - x0) + y0)

        rectangle = [[(up_left_x1, up_left_y1), (down_left_x1, down_left_y1)],
                     [(down_left_x1, down_left_y1), (down_right_x1, down_right_y1)],
                     [(down_right_x1, down_right_y1), (up_right_x1, up_right_y1)],
                     [(up_right_x1, up_right_y1), (up_left_x1, up_left_y1)]]

    def rotateFace(self, angle, image):
        """
        Функция для поворота лица на заданный угол и, соответственно, построения прямоугольника.
        Для правильного выреза необходимого окна заметим, что центр поворота - инвариант. Следовательно, если мы отступим
        половину длины прямоугольника вверх и влево, то, очеивдно, получим необходимые границы лица.
        :return:
        extreme_points -- Критические точки лица (Словарь).
        dist_x -- Высота Прямоугольника.
        dist_y -- Ширина Прямоугольника.
        """

        def point_rotation_x(x, y, x0, y0, angle):
            """
            Function to rotate a point (x,y) around (x0,y0) by a given angle
            :param x:
            :param y:
            :param x0:
            :param y0:
            :param angle:
            :return:
            """
            return -1 * math.sin(angle) * (y - y0) + math.cos(angle) * (x - x0) + x0

        def point_rotation_y(x, y, x0, y0, angle):
            """
            Function to rotate a point (x,y) around (x0,y0) by a given angle
            :param x:
            :param y:
            :param x0:
            :param y0:
            :param angle:
            :return:
            """
            return math.cos(angle) * (y - y0) + math.sin(angle) * (x - x0) + y0

        def rotation_matrix(angle: float, x: list, y: list, x0: float, y0: float) -> list:
            """A function that rotates the specified segments by a given angle (angle) using a rotation matrix.

            Keyword arguments:
                angle -- Landmark list.
                x -- Points of the face along the abscissa.
                y -- Points of the face along the ordinates.
                x0 -- The point around which the turn occurs.
                y0 -- The point around which the turn occurs.
            """

            # Left Top Point of Rectangle.
            up_left_x1 = int(-1 * math.sin(angle) * (min(y) - y0) + math.cos(angle) * (min(x) - x0) + x0)
            up_left_y1 = int(math.cos(angle) * (min(y) - y0) + math.sin(angle) * (min(x) - x0) + y0)

            # Right Upper point of the rectangle.
            up_right_x1 = int(-1 * math.sin(angle) * (min(y) - y0) + math.cos(angle) * (max(x) - x0) + x0)
            up_right_y1 = int(math.cos(angle) * (min(y) - y0) + math.sin(angle) * (max(x) - x0) + y0)

            # Right Lower Point of Rectangle.
            down_right_x1 = int(-1 * math.sin(angle) * (max(y) - y0) + math.cos(angle) * (max(x) - x0) + x0)
            down_right_y1 = int(math.cos(angle) * (max(y) - y0) + math.sin(angle) * (max(x) - x0) + y0)

            # Left Lower Point of the Rectangle.
            down_left_x1 = int(-1 * math.sin(angle) * (max(y) - y0) + math.cos(angle) * (min(x) - x0) + x0)
            down_left_y1 = int(math.cos(angle) * (max(y) - y0) + math.sin(angle) * (min(x) - x0) + y0)

            return [[(up_left_x1, up_left_y1), (down_left_x1, down_left_y1)],
                    [(down_left_x1, down_left_y1), (down_right_x1, down_right_y1)],
                    [(down_right_x1, down_right_y1), (up_right_x1, up_right_y1)],
                    [(up_right_x1, up_right_y1), (up_left_x1, up_left_y1)]]

        x_points = self.x_points
        y_points = self.y_points
        x0 = self.x0
        y0 = self.y0

        x_rotate_points = point_rotation_x(x_points, y_points, x0, y0, -1 * angle)
        y_rotate_points = point_rotation_y(x_points, y_points, x0, y0, -1 * angle)

        rectangle = rotation_matrix(angle, x_rotate_points, y_rotate_points, x0, y0)

        cv2.line(image, rectangle[0][0], rectangle[0][1], (127, 255, 0), 4)
        cv2.line(image, rectangle[1][0], rectangle[1][1], (127, 255, 0), 4)
        cv2.line(image, rectangle[2][0], rectangle[2][1], (127, 255, 0), 4)
        cv2.line(image, rectangle[3][0], rectangle[3][1], (127, 255, 0), 4)

        extreme_points = {'up_Left': rectangle[0][0],
                          'up_Right': rectangle[2][1],
                          'down_Left': rectangle[0][1],
                          'down_Right': rectangle[1][1]}

        self.extremums = extreme_points

        return extreme_points, \
               math.hypot(extreme_points['up_Left'][0] - extreme_points['down_Left'][0],
                          extreme_points['up_Left'][1] - extreme_points['down_Left'][1]), \
               math.hypot(extreme_points['up_Left'][0] - extreme_points['up_Right'][0],
                          extreme_points['up_Left'][1] - extreme_points['up_Right'][1])

    def getSortPoints(self, extreme_points: dict):
        """"""
        sort_up_points = [extreme_points['up_Left'], extreme_points['up_Right']]
        sort_up_points.sort(key=lambda elem: elem[1])

        sort_down_points = [extreme_points['down_Left'], extreme_points['down_Right']]
        sort_down_points.sort(key=lambda elem: elem[1])

        return [sort_up_points, sort_down_points[::-1]]

    def getExtreme_Points(self):
        """
        {'up_Left': rectangle[0][0],
        'up_Right': rectangle[2][1],
        'down_Left': rectangle[0][1],
        'down_Right': rectangle[1][1]}

        :return:
        """
        return self.extremums
