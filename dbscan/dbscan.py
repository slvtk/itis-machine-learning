import numpy as np

RED = (140, 0, 0)
GREEN = (0, 140, 0)
BLUE = (0, 0, 140)
BLACK = (0, 0, 0)


class DBScan:
    def dbs(self, points, eps):
        labels = [0] * len(points)
        clust_id = 0
        for i in range(0, len(points)):
            if labels[i] != 0:
                continue
            near_points = self.nearest(points, i, eps)
            if len(near_points) < 1:
                labels[i] = -1
            else:
                clust_id += 1
                labels[i] = clust_id
                i = 0
                while i < len(near_points):
                    point = near_points[i]
                    if labels[point] == -1:
                        labels[point] = clust_id
                    elif labels[point] == 0:
                        labels[point] = clust_id
                        point_near = self.nearest(points, point, eps)
                        if len(point_near) >= 1:
                            near_points = near_points + point_near
                    i += 1

        return labels

    def nearest(self, points, idx, eps):
        near = []
        for point_idx in range(0, len(points)):
            if np.linalg.norm(points[idx] - points[point_idx]) < eps:
                near.append(point_idx)
        return near

    def point_color(self, list_col):
        if list_col == 1: return RED
        if list_col == 2: return GREEN
        if list_col == 3: return BLUE
        if list_col == 4: return BLACK
        return 125, 125, 125
