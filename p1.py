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

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]):
  """
  Altera el rango dinámico de la imagen.
  """
  if inRange == []:
    min_in = np.min(inImage)
    max_in = np.max(inImage)
  else :
    min_in = inRange[0]
    max_in = inRange[1]
  min_out = outRange[0]
  max_out = outRange[1]
  return min_out + (((max_out - min_out)*inImage - min_in)/(max_in - min_in))

def main():
  #
  #  Test de adjustIntensity
  #
  image = read_img("./prueba/circles.png")
  show(image)
  image2 = adjustIntensity(image, [0, 0.5], [0, 1])
  show(image2)

if __name__ == "__main__":
  main()