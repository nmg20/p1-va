import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes
import p1

# Funciones auxiliares

def read_img(path):
  return cv.imread(path, cv.IMREAD_GRAYSCALE)/255.

def show(image):
  if (image.shape[1]>300):
    image = imutils.resize(image,width=300)
  cv.imshow("", image)
  cv.waitKey(0)
  cv.destroyAllWindows()

def show(img1, img2):
  height, width = img1.shape[:2]  #Suponemos que ambas imágenes tienen el mismo tamaño (Original/Modificada)
  if (width>300):
    img1 = imutils.resize(img1,width=300) #Resize proporcional sólo para mostrar las imágenes
    img2 = imutils.resize(img2,width=300)
  # print("Height: ",height,"\tWidth: ",width)
  pack = np.concatenate((img1, img2), axis=1)
  cv.imshow("", pack)
  cv.waitKey(0)
  cv.destroyAllWindows()

###########################################################

def 


def edgeCanny(inImage, sigma, tlow, thigh):


  return null


#####

def main():
  # image = read_img("./imagenes/morphology/closed.png")
  # image = read_img("./imagenes/grad7.png")
  image = read_img("./imagenes/lena.png")

  gx, gy = gradientImage(image, "CentralDiff")
  gx = adjustIntensity(gx, [], [0,1])
  gy = adjustIntensity(gy, [], [0,1])
  show(image, gx)
  show(image, gy)

if __name__ == "__main__":
  main()

