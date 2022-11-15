import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes
import p1

def direccion(gx, gy):
  m,n = gx.shape()
  direcion = np.zeros((m,n), dtype='float32')
  for i in range(m):
    for j in range(n):
      if (gy[i,j]==0):
        direccion[i,j]=0
      else:
        direccion[i,j]=np.arctan(gx[i,j]/gy[i,j])
  return direccion

def mejora(img, sigma):
  suavizado = p1.gaussianFilter(img, sigma)
  m, n = img.shape
  for i in range(m):
    for j in range(n):
      


def edgeCanny(inImage, sigma, tlow, thigh):


  return null


#####

def main():
  # image = p1.read_img("./imagenes/morphology/closed.png")
  # image = p1.read_img("./imagenes/grad7.png")
  image = p1.read_img("./imagenes/lena.png")

  gx, gy = p1.gradientImage(image, "CentralDiff")
  gx = p1.adjustIntensity(gx, [], [0,1])
  gy = p1.adjustIntensity(gy, [], [0,1])
  p1.show(image, gx)
  p1.show(image, gy)

if __name__ == "__main__":
  main()

