import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes

# Dudas:
# - EqualizeIntensity -> usar numpy.hist o algo asi
# - FilterImage -> Omitir bordes o añadir padding?
# - GaussFilter1D -> 0 a N o -centro a centro?
#

# Funciones auxiliares

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

#

def medianFilter(inImage, filterSize): # [1]
  """
  Suaviza una imagen mediante un filtro de medianas bidimensional. 
  El tamaño del kernel del filtro viene dado por filterSize.
  """
  m, n = np.shape(inImage)
  outImage = inImage # Se usa la imagen original para evitar bordes en negro
  centro = filterSize//2
  for x in range(centro,m-centro,1):
    for y in range(centro,n-centro,1):
      window = inImage[(x-centro)::(x+centro),(y-centro)::(y+centro)]
      outImage[x,y] = np.sum(window)/(len(window)*len(window[0]))
  return outImage

#####

def main():
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # image = read_img("./imagenes/77.png")
  # image = read_img("./imagenes/blob55.png")
  # image = read_img("./imagenes/point55.png")
  # image = read_img("./imagenes/x55.png")
  image = read_img("./imagenes/salt77.png")

  show(image)
  image2 = medianFilter(image, 5)
  show(image2)

if __name__ == "__main__":
  main()

