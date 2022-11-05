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

############### Versiones con Padding ###############

### Erosión ###

def erode(img, SE, center=[]):
  m, n = np.shape(img)
  p, q = np.shape(SE)
  out = np.zeros((m,n), dtype='float32')
  padH, padV = p//2, q//2
  if center==[]:
    center=[padH, padV]
  pad = cv.copyMakeBorder(img, padH, padH, padV, padV, cv.BORDER_CONSTANT)
  for x in range(padH, m+padH, 1): #Recorremos la distancia de la imagen
    for y in range(padV, n+padV, 1): # original dentro de la paddeada
      limar = x-center[0]
      limab = x+p-center[0]
      limizq = y-center[1]
      limder = y+q-center[1]
      window = pad[limar:limab, limizq:limder]
      out[x-padH,y-padV]=img[x-padH, y-padV]
      # Se comprueba que se erosiona el píxel
      for i in range(p):
        for j in range(q):
          if window[i][j]==0 and SE[i][j]==1:
            out[x-padH, y-padV]=0
  return out

### Dilatación ###

def dilate(img, SE, center=[]):
  m, n = np.shape(img)
  p, q = np.shape(SE)
  out = np.zeros((m,n), dtype='float32')
  padH, padV = p//2, q//2
  if center==[]:
    center=[padH, padV]
  pad = cv.copyMakeBorder(img, padH, padH, padV, padV, cv.BORDER_CONSTANT)
  for x in range(padH, m+padH, 1): #Recorremos la distancia de la imagen
    for y in range(padV, n+padV, 1): # original dentro de la paddeada
      limar = x-center[0]
      limab = x+p-center[0]
      limizq = y-center[1]
      limder = y+q-center[1]
      window = pad[limar:limab, limizq:limder]
      out[x-padH,y-padV]=img[x-padH, y-padV]
      # Se comprueba que se erosiona el píxel
      for i in range(p):
        for j in range(q):
          if window[i][j]==1 and SE[i][j]==1:
            out[x-padH, y-padV]=1
  return out

############### Versiones sin padding ###############

# def erode(img, SE, center=[]):
#   m, n = np.shape(img)
#   p, q = np.shape(SE)
#   out = np.zeros((m,n), dtype='float32')
#   if center==[]:
#     center=[p//2, q//2]
#   for x in range(m): #Recorremos la distancia de la imagen
#     for y in range(n): # original dentro de la paddeada
#       limar = max(0,x-center[0])
#       limab = min(m,x+p-center[0])
#       limizq = max(0,y-center[1])
#       limder = min(n,y+q-center[1])
#       window = img[limar:limab, limizq:limder]
#       out[x,y]=img[x,y]
#       SEAux = SE[limar:limab, limizq:limder]
#       # Se comprueba que se erosiona el píxel
#       for i in range(p):
#         for j in range(q):
#           if window[i][j]==0 and SEAux[i][j]==1:
#             out[x, y]=0
#   return out

#####

def main():
  # image = read_img("./imagenes/morphology/diagonal.png")
  # image = read_img("./imagenes/morphology/blob.png")
  # image = read_img("./imagenes/morphology/a34.png")
  image = read_img("./imagenes/morphology/ex.png")

  ### Erode ###

  SE = [[0,1,0],[1,1,1],[0,1,0]]
  image2 = erode(image, SE, [])
  show2(image, image2)

  ### Dilate ###

  # SE = [[0,1,0],[1,1,1],[0,1,0]]
  # image2 = dilate(image, SE, [1,1])
  # show2(image, image2)

if __name__ == "__main__":
  main()

