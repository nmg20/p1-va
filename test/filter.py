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

####### Versión in bordes #######

def filterB(inImage, kernel):
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

####### Versión con padding ######

def filterP(inImage, kernel):
  m, n = np.shape(inImage) # Tamaño de la imagen
  p, q = np.shape(kernel) # Tamaño del kernel
  a = p // 2
  b = q // 2
  outImage = np.zeros((m-(a-1),n-(b-1)), dtype='float32') # Img resultado de menor tamaño
  img = cv.copyMakeBorder(inImage,a,a,b,b,cv.BORDER_CONSTANT)
  for x in range(a, m-a, 1):
    for y in range(b, n-b, 1):
      window = inImage[(x-a):(x+p-a),(y-b):(y+q-b)]
      outImage[x,y] = (window * kernel).sum()
  return outImage

#####

def main():
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # image = read_img("./imagenes/77.png")
  # image = read_img("./imagenes/blob55.png")
  # image = read_img("./imagenes/point55.png")
  # image = read_img("./imagenes/x55.png")
  # image = read_img("./imagenes/salt77.png")
  image = read_img("./imagenes/salt99.png")

  show(image)
  # kernel = [[0,1,0],[1,1,1],[0,1,0]]
  kernel = [[0,0.1,0],[0.1,0.1,0.1],[0,0.1,0]]
  # kernel = [[0,0.5,0],[0.5,0.5,0.5],[0,0.5,0]]
  image2 = filterB(image, kernel)
  image2 = filterP(image, kernel)
  show(image2)

if __name__ == "__main__":
  main()

