import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

M = 1.5
E = 0.001


class Cluster:
    def __init__(self, centroids_x, centroids_y):
        self.points_x = []
        self.points_y = []
        self.centroids_x = centroids_x
        self.centroids_y = centroids_y

    def append_point(self, x, y):
        self.points_x.append(x)
        self.points_y.append(y)

    def extend_point(self, x, y):
        self.points_x.extend(x)
        self.points_y.extend(y)

    def clear_points(self):
        self.points_x = []
        self.points_y = []

    def mean(self):
        self.centroids_x = sum(self.points_x) / len(self.points_x)
        self.centroids_y = sum(self.points_y) / len(self.points_y)

    def set_centr(self, x, y):
        self.centroids_x = x
        self.centroids_y = y


def fill_zero(k, p):
    u_new = []
    u_pre = []
    for i in range(k):
        u_new.append([float('inf')] * p)
    for i in range(k):
        u_pre.append([0] * p)
    return [u_new, u_pre]


def draw(clusters, matrix, points):
    color = ['r', 'b', 'g', 'c', 'm', 'k', 'y', 'lime', 'teal', 'navy', 'plum', 'pink', 'cyan', 'slategray', 'peru',
             'brown']
    i = 0
    for cl in clusters:
        plt.scatter(cl.centroids_x, cl.centroids_y, color=color[i], marker='x')
        plt.scatter(cl.points_x, cl.points_y, color=color[i])
        i += 1
    plt.show()


def map_to_array(clusters):
    x_c = []
    y_c = []
    for cl in clusters:
        x_c.append(cl.centroids_x)
        y_c.append(cl.centroids_y)
    return [x_c, y_c]


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def random_points(n, filename):
    x = np.random.randint(0, 100, n)
    y = np.random.randint(0, 100, n)
    pd.DataFrame([x, y]).to_csv(filename)


def generate_points_file(points_count, filename):
    x = np.random.randint(0, 100, points_count)
    y = np.random.randint(0, 100, points_count)
    with open(filename, 'w') as file:
        file.write('x,y\n')
        for item_x, item_y in zip(x, y):
            file.write(f'{item_x},{item_y}\n')


def read_csv(filename):
    return pd.read_csv(filename)


def centroids(points, k):
    x_centr = points['x'].mean()
    y_centr = points['y'].mean()
    R = dist(x_centr, y_centr, points['x'][0], points['y'][0])
    for i in range(len(points)):
        R = max(R, dist(x_centr, y_centr, points['x'][i], points['y'][i]))
    x_c, y_c = [], []
    for i in range(k):
        x_c.append(x_centr + R * np.cos(2 * np.pi * i / k))
        y_c.append(y_centr + R * np.sin(2 * np.pi * i / k))
    return [x_c, y_c]


def nearest_centroid(points, centroids):
    clusters = [Cluster(x_c, y_c) for x_c, y_c in zip(centroids[0], centroids[1])]
    indx = -1
    for x, y in zip(points['x'], points['y']):
        r = float('inf')
        for i, cl in enumerate(clusters):
            if r > dist(x, y, cl.centroids_x, cl.centroids_y):
                r = dist(x, y, cl.centroids_x, cl.centroids_y)
                indx = i
        if indx >= 0:
            clusters[indx].append_point(x, y)
    return clusters


def recalculate_centroid(clusters):
    new_clusters = []
    for cl in clusters:
        new_cl = Cluster(cl.centroids_x, cl.centroids_y)
        new_cl.extend_point(cl.points_x, cl.points_y)
        new_clusters.append(new_cl)
    for cl in new_clusters:
        if len(cl.points_x) != 0:
            cl.mean()
    return new_clusters


def calculate_centr_cluster(clusters, u_new):
    j = 0
    new_clusters = []
    for cl in clusters:
        new_cl = Cluster(cl.centroids_x, cl.centroids_y)
        new_cl.extend_point(cl.points_x, cl.points_y)
        new_clusters.append(new_cl)
    for cl in new_clusters:
        i = 0
        s = 0
        for l in range(len(cl.points_x)):
            s += u_new[j][l] ** M
        s_cl_x = 0
        s_cl_y = 0
        for x, y in zip(cl.points_x, cl.points_y):
            s_cl_x += ((u_new[j][i]) ** M) * x
            s_cl_y += ((u_new[j][i]) ** M) * y
            i += 1
        cl.set_centr(s_cl_x / s, s_cl_y / s)
        j += 1
    return new_clusters


def is_not_finished(u_new, u_pre):
    for i in range(len(u_pre)):
        for j in range(len(u_pre[0])):
            if abs(u_new[i][j] - u_pre[i][j]) < E:
                return False
    return True


def cal_membership_coef(centroids, points):
    matrix = []
    for i in range(len(centroids[0])):
        matrix.append([])
    for i in range(len(points['x'])):
        dist_sum = 0.0
        dist_sum = np.float_(dist_sum)
        m = np.float_(M)
        for j in range(len(centroids[0])):
            dist_sum += dist(points['x'][i], points['y'][i], centroids[0][j], centroids[1][j]) ** 2 / (1 - m)
        for j in range(len(centroids[0])):
            prob = (dist(points['x'][i], points['y'][i], centroids[0][j], centroids[1][j]) ** 2 / (1 - m))
            matrix[j].append(prob / dist_sum)
    return matrix


def c_l(clusters):
    r = []
    for cl in clusters:
        for x, y in zip(cl.points_x, cl.points_y):
            r.append(dist(cl.centroids_x, cl.centroids_y, x, y))
    return sum(r)


def d_l(s0, s1, s2):
    if abs(s0 - s1) != 0:
        return abs(s1 - s2) / abs(s0 - s1)
    else:
        return float('inf')


def c_means(points, k, is_show, u):
    cntds = centroids(points, k)
    clusters = nearest_centroid(points, cntds)
    new_clusters = []
    if (is_show):
        draw(clusters, u[0], points)
    while is_not_finished(u[0], u[1]):
        old_clusters = nearest_centroid(points, cntds)
        new_clusters = recalculate_centroid(old_clusters)
        cntds = map_to_array(new_clusters)
        u[1] = u[0]
        u[0] = cal_membership_coef(cntds, points)
        new_clusters = calculate_centr_cluster(new_clusters, u[0])
        cntds = map_to_array(new_clusters)
        if (is_show):
            draw(new_clusters, u[0], points)
    return new_clusters


if __name__ == "__main__":
    n = 100
    k_max = 15
    k = 3
    c_ls = []
    d_ls = []
    points = read_csv('data.csv')
    for i in range(k, k_max):
        u = fill_zero(i, len(points))
        clusters = c_means(points, i, False, u)
        c_ls.append(c_l(clusters))
    dl = float('inf')
    k = 0
    for i in range(0, len(c_ls) - 2):
        d_ls.append(d_l(c_ls[i], c_ls[i + 1], c_ls[i + 2]))
        if dl > d_l(c_ls[i], c_ls[i + 1], c_ls[i + 2]):
            dl = d_l(c_ls[i], c_ls[i + 1], c_ls[i + 2])
            k = i + 1

    u = fill_zero(k, len(points))
    print(k)
    c_means(points, k, True, u)
