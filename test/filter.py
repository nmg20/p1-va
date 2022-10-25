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
  """
  Muestra dos imágenes una al lado de otra.
  """
  height, width = img1.shape[:2]  #Suponemos que ambas imágenes tienen el mismo tamaño (Original/Modificada)
  if (width>300):
    img1 = imutils.resize(img1,width=300) #Resize proporcional sólo para mostrar las imágenes
    img2 = imutils.resize(img2,width=300)
  # print("Height: ",height,"\tWidth: ",width)
  pack = np.concatenate((img1, img2), axis=1)
  cv.imshow("", pack)
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
  outImage = np.zeros((m,n), dtype='float32') # Img resultado de menor tamaño
  img = cv.copyMakeBorder(inImage,a,a,b,b,cv.BORDER_CONSTANT)
  for x in range(a, m+a, 1):
    for y in range(b, n+b, 1):
      window = img[(x-a):(x+p-a),(y-b):(y+q-b)]
      # print("Window:\t(",x-a,"---",x+p-a,")\n\t(",y-b,"---",y+q-b)
      outImage[x-a,y-b] = (window * kernel).sum()
  return outImage

#####

def main():
  # image = read_img("./imagenes/circles.png")
  image = read_img("./imagenes/circles1.png")
  # image = read_img("./imagenes/77.png")
  # image = read_img("./imagenes/blob55.png")
  # image = read_img("./imagenes/point55.png")
  # image = read_img("./imagenes/x55.png")
  # image = read_img("./imagenes/salt77.png")
  # image = read_img("./imagenes/salt99.png")

  # kernel = [[0,1,0],[1,1,1],[0,1,0]]
  # kernel = [[0,0.1,0],[0.1,0.1,0.1],[0,0.1,0]]
  kernel = [[0,0.5,0],[0.5,0.5,0.5],[0,0.5,0]]
  # image2 = filterB(image, kernel)
  image2 = filterP(image, kernel)
  # show(image)
  # show(image2)
  show2(image,image2)

if __name__ == "__main__":
  main()

