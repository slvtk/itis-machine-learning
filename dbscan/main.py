import numpy as np
import pygame

from dbscan import DBScan

WHITE = (255, 255, 255)


def main():
    pygame.init()
    dbcan = DBScan()
    points = []

    window = pygame.display.set_mode((1280, 720))
    window.fill(WHITE)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                points.append(pygame.mouse.get_pos())
                window.fill(WHITE)
                prediction = dbcan.dbs(np.array(points), 40)
                for point, cluster in zip(points, prediction):
                    pygame.draw.circle(window, dbcan.point_color(cluster), point, 10)

        pygame.display.update()


if __name__ == '__main__':
    main()
