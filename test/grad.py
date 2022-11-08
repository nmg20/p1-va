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
#####

# Roberts: (Gx = [[-1,0],[0,1]], Gy=[[0,-1],[1,0]])
# CentralDiff: 
#           
# Prewitt: (Gx = [[-1,0,1],[-1,0,1],[-1,0,1]],
#           Gy = [[-1,-1,-1],[0,0,0],[1,1,1]])
# Sobel:   (Gx = [[-1,0,1],[-1,0,1],[-1,0,1]],
#           Gy = [[-1,-2,-1],[0,0,0],[1,2,1]])
#

def gradientImage(inImage, operator):
  """
  Según el operador escoger una matriz a modo de SE
    -> aplicarla convolucionalmente por la imagen
  Operator => ["roberts","prewitt","sobel"]
  """
  m, n = inImage.shape
  gx, gy = np.zeros((m,n), dtype='float32'), np.zeros((m,n), dtype='float32')
  if operator == "roberts":
    maskx, masky = np.array([[-1,0],[0,1]]), np.array([[0,-1],[1,0]])
  elif operator == "prewitt":
    maskx, masky = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
      np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
  elif operator == "sobel":
    maskx, masky = np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),
      np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
  p, q = maskx
  for x in range(m):
    for y in range(n):

  return [gx, gy]

#####

def main():
  image = read_img("./imagenes/morphology/closed.png")

  ### Erode ###

  # SE = [[0,1,0],[1,1,1],[0,1,0]]
  # seeds = [[1,1],[4,2]]
  # image2 = erode(image, SE, [])
  # show2(image, image2)

if __name__ == "__main__":
  main()

