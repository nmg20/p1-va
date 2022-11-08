import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes

# Necesario:
# - Operación básica de llenar una región con el SE
# - Recorrer una región según una seed
# - Recorrer región si la seed no es el primer píxel de la región
# - Registrar regiones llenadas
# - Aplicar para varias seeds


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

# def filled()

def fillWin(window, SE, x, y):
  """
  Llena una zona respectiva al tamaño del SE. 
    ->Operación atómica -> se aplica convolutivamente.
  """
  m, n = window.shape
  result = np.zeros((m,n), dtype='float32')
  filled = []
  for i in range(m):
    for j in range(n):
      if (window[i][j]==0 and SE[i][j]==1):
        filled.append([i+x-1,j+y-1])
      result[i][j]=(window[i][j]==1 or SE[i][j]==1)
  return result, filled

# def fillReg(reg, SE):
#   """
#   Aplica fillWin a toda una región cerrada.
#   """

def fill(inImage, seeds, SE=[], center=[]):
  if SE == []:
    SE = np.array([[0,1,0],[1,1,1],[0,1,0]])
  m, n = inImage.shape
  p,q = SE.shape
  a, b = p//2, q//2
  if center == []:
    center=[a,b]
  # outImage = np.zeros((m,n), dtype='float32')
  outImage = inImage.copy()
  filled = []
  # pad = cv.copyMakeBorder(inImage, p//2,p//2,q//2,q//2,cv.BORDER_CONSTANT)
  for (sx, sy) in seeds:
    for x in range(sx,m,1):
      for y in range(sy,n,1):
        limar = max(0,x-(a))
        limab = min(m,x+p-(a))
        limizq = max(0,y-(b))
        limder = min(n,y+q-(b))
        window = inImage[limar:limab, limizq:limder]
        r, f = fillWin(window, SE, x, y)
        outImage[limar:limab, limizq:limder]=r
        filled = filled + f
  print("Filled:",filled)
  return outImage, filled


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

