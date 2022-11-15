import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes
import p1

def medianFilter(inImage, filterSize):
  m, n = np.shape(inImage)
  outImage = np.zeros((m,n), dtype='float32')
  centro = filterSize//2
  img = cv.copyMakeBorder(inImage,centro,centro,centro,centro,cv.BORDER_CONSTANT)
  for x in range(m):
    for y in range(n):
      # limizq = max(0,y-centro)
      # limder = min(n,y+filterSize-centro)
      # limar = max(0,x-centro)
      # limab = min(m,x+filterSize-centro)
      # window = inImage[limar:limab,limizq:limder]
      window = img[x:x+filterSize,y:y+filterSize]
      # outImage[x,y] = np.sum(window)/(len(window)*len(window[0]))
      outImage[x,y] = np.median(window)
      # print("---------------------")
      # print("Posición: (",x,",",y,"): ",outImage[x-centro,y-centro])
      # # print("\t",limar,"\n",limizq,"\t\t",limder,"\n\t",limab)
      # print("Límites:\n\t(",limizq,",",limar,")------(",limder,",",limar,")")
      # print("\t|\t\t\t|\n\t(",limizq,",",limab,")-------(",limder,",",limab,")")
  return outImage

#####

def main():
  # image = p1.read_img("./imagenes/circles.png")
  # image = p1.read_img("./imagenes/circles1.png")
  # image = p1.read_img("./imagenes/x55.png")
  # image = read_img("./imagenes/white55.png")
  # image = p1.read_img("./imagenes/r44.png")
  # image = p1.read_img("./imagenes/cosa55.png")
  # image = p1.read_img("./imagenes/salt77.png")
  # image = p1.read_img("./imagenes/salt99.png")
  # image = p1.read_img("./imagenes/salt11.png")
  image = p1.read_img("./imagenes/saltgirl.png")

  image2 = medianFilter(image, 3)
  p1.show(image, image2)

if __name__ == "__main__":
  main()

