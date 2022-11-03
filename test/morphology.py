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

# def erode(inImage, SE, center=[]):
#   SE = np.array(SE)
#   m, n = np.shape(inImage)
#   p, q = np.shape(SE)
#   outImage = np.zeros((m,n), dtype='float32')
#   if not center:
#     center = [p//2,q//2] 
#   image = cv.copyMakeBorder(inImage,p,p,q,q,cv.BORDER_CONSTANT)
#   for x in range(m):
#     for y in range(n):
#       limizq = max(0,y-center[1])
#       limder = min(n,y+q-center[1])
#       limar = max(0,x-center[0])
#       limab = min(m,x+p-center[0])
#       # window = inImage[limar:limab, limizq:limder]
#       window = image[limar:limab, limizq:limder]
#       SE=SE[center[0]:p, center[1]:q]
#       # Se comprueba si hay que erosionar -> 1 en el EE, 0 en la sección analizada
#       eroded = 0
#       for i in range(p-1):
#         for j in range(q-1):
#           if SE[i,j]==1 and window[i,j]==0:
#             eroded = eroded + 1
#       if (np.sum(window)>0 and eroded==0):
#         outImage[x,y] = image[x,y]
#   return outImage

def erode(img, SE, center=[]):
  m, n = np.shape(img)
  p, q = np.shape(SE)
  out = np.zeros((m,n), dtype='float32')
  padH, padV = p//2, q//2
  center = center if center else [padH, padV]
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

# def erode(img, se, center):
#   w, h = img.shape
#   p, q = se.shape
#   if center==[]:
#     center=[p//2, q//2]
#   out = np.zeros((m,n), dtype='float32')



#####

def main():
  # image = read_img("./imagenes/morphology/diagonal.png")
  # image = read_img("./imagenes/morphology/blob.png")
  image = read_img("./imagenes/morphology/a34.png")

  # show(image)
  SE = [[1,1]]
  image2 = erode(image, SE, [0,1])
  # show(image)
  # show(image2)
  show2(image, image2)

if __name__ == "__main__":
  main()

