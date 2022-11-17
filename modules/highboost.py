import numpy as np
import cv2 as cv
import sys
sys.path.append('../')
import p1
def highBoost(inImage, A, method, param):
  m, n = np.shape(inImage)
  realzado = np.zeros((m,n), dtype='float32')
  if method=='gaussian':
    suavizado = p1.gaussianFilter(inImage, param)
  elif method=='median':
    suavizado = p1.medianFilter(inImage, param)
  for x in range(0,m,1):
    for y in range(0,n,1):
        realzado[x,y] = A*inImage[x,y]-suavizado[x,y]
  return realzado

#####

def main():
  # image = p1.read_img("../imagenes/circles.png")
  # image = p1.read_img("../imagenes/circles1.png")
  # image = p1.read_img("../imagenes/77.png")
  # image = p1.read_img("../imagenes/blob55.png")
  # image = p1.read_img("../imagenes/point55.png")
  # image = p1.read_img("../imagenes/x55.png")
  image = p1.read_img("../imagenes/salt99.png")


  # image2 = highBoost(image, 2, 'gaussian', 0.5)
  image2 = highBoost(image, 2, 'median', 3)
  # p1.show(image)
  p1.show(image,image2)

if __name__ == "__main__":
  main()

