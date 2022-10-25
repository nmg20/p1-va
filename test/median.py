import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes

# Funciones auxiliares

def read_img(path):
  return cv.imread(path, cv.IMREAD_GRAYSCALE)/255.

def show(image):
  if (image.shape[1]>300):
    image = imutils.resize(image,width=300)
  cv.imshow("", image)
  cv.waitKey(0)
  cv.destroyAllWindows()

def show2(img1, img2):
  height, width = img1.shape[:2]  #Suponemos que ambas imágenes tienen el mismo tamaño (Original/Modificada)
  if (width>300):
    img1 = imutils.resize(img1,width=300) #Resize proporcional sólo para mostrar las imágenes
    img2 = imutils.resize(img2,width=300)
  # print("Height: ",height,"\tWidth: ",width)
  pack = np.concatenate((img1, img2), axis=1)
  cv.imshow("", pack)
  cv.waitKey(0)
  cv.destroyAllWindows()

#

def medianFilter(inImage, filterSize):
  m, n = np.shape(inImage)
  outImage = np.zeros((m,n), dtype='float32')
  centro = filterSize//2
  for x in range(m):
    for y in range(n):
      limizq = max(0,y-centro)
      limder = min(n,y+filterSize-centro)
      limar = max(0,x-centro)
      limab = min(m,x+filterSize-centro)
      window = inImage[limar:limab,limizq:limder]
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
  # image = read_img("./imagenes/circles.png")
  # image = read_img("./imagenes/circles1.png")
  # image = read_img("./imagenes/x55.png")
  # image = read_img("./imagenes/white55.png")
  # image = read_img("./imagenes/r44.png")
  # image = read_img("./imagenes/cosa55.png")
  # image = read_img("./imagenes/salt77.png")
  # image = read_img("./imagenes/salt99.png")
  # image = read_img("./imagenes/salt11.png")
  image = read_img("./imagenes/saltgirl.png")

  # show(image)
  image2 = medianFilter(image, 3)
  # show(image)
  # show(image2)
  show2(image, image2)

if __name__ == "__main__":
  main()

