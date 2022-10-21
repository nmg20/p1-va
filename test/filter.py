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

def filterImage(inImage, kernel):
  """
  Aplica un filtro mediante convolución de un kernel sobre una imagen.
  - kernel = array/matriz de coeficientes
  """
  m, n = np.shape(inImage) # Tamaño de la imagen
  p, q = np.shape(kernel) # Tamaño del kernel
  a = p // 2
  b = q // 2
  outImage = np.zeros((m-(a-1),n-(b-1)), dtype='float32') # Img resultado de menor tamaño
  for x in range(a, m-a, 1):
    for y in range(b, n-b, 1):
      window = inImage[(x-a):(x+p-a),(y-b):(y+q-b)]
      outImage[x,y] = (window * kernel).sum()
    #   print(outImage[x,y], end="  ")
    # print("\n")
      
  return outImage

#####

def main():
  # image = read_img("../prueba/circles.png")
  image = read_img("../prueba/circles1.png")
  # image = read_img("../prueba/77.png")
  # image = read_img("../prueba/blob55.png")
  # image = read_img("../prueba/point55.png")
  # image = read_img("../prueba/x55.png")
  # image = read_img("../prueba/white_dot55.png")

  show(image)
  # kernel = [[0,1,0],[1,1,1],[0,1,0]]
  kernel = [[0,0.1,0],[0.1,0.1,0.1],[0,0.1,0]]
  # kernel = [[0,0.5,0],[0.5,0.5,0.5],[0,0.5,0]]
  image2 = filterImage(image, kernel)
  show(image2)

if __name__ == "__main__":
  main()

