import pygame
import sklearn.svm as svm
import numpy as np


def draw_circle(coordinates, classification, current_event, current_class):
    color = BLUE if current_class == 1 else RED
    pygame.draw.circle(window, color, current_event.pos, radius)
    coordinates.append(current_event.pos)
    classification.append(current_class)


BLUE = (0, 0, 255)
RED = (255, 0, 0)
WHITE = (255, 255, 255)
radius = 5
line_width = 2
button_left = 1
button_rigth = 3

pygame.init()

window = pygame.display.set_mode((1280, 720))
window.fill(WHITE)
pygame.display.update()

coordinates = []
classification = []

game = True
while game:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == button_left:
                draw_circle(coordinates, classification, event, 1)
            elif event.button == button_rigth:
                draw_circle(coordinates, classification, event, 2)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            clf = svm.SVC(kernel='linear', C=1.0)
            clf.fit(coordinates, classification)

            a = clf.coef_[0][0]
            b = clf.coef_[0][1]
            c = clf.intercept_

            x = np.linspace(0, 1280, 2)
            y = (-1 * c - a * x) / b

            pygame.draw.line(window, (0, 0, 0), [x[0], y[0]], [x[1], y[1]], line_width)

        pygame.display.update()

pygame.quit()