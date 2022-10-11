import numpy as np
import cv2 as cv

def read_img(path):
  """
  Devuelve una imagen en punto flotante a partir de su path.
  return - array (imagen)
  """
  return cv.imread(path, cv.IMREAD_GRAYSCALE)/255.

def show(image):
  """
  Muestra la imagen proporcionada por pantalla.
  """
  cv.imshow("", image)
  cv.waitKey(0)
  cv.destroyAllWindows()