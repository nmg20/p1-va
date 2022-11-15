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

def dilate(inImage, SE, center=[]):
  """
  Aplica el operador morfológico de dilatación.
    - inImage: imagen -> conjuntos de posiciones con 1s.
    - SE: elemento estructurante.
    - center: origen del SE. Se asume que el [0, 0] es la esquina
        superior izquierda. Si está vacío, el centro es ([P/2]+1, [Q/2]+1).
  """
  m, n = np.shape(inImage)
  p, q = np.shape(SE)
  outImage = np.zeros((m,n), dtype='float32')
  padH, padV = p//2, q//2
  if center==[]:
    center=[padH, padV]
  pad = cv.copyMakeBorder(inImage, padH, padH, padV, padV, cv.BORDER_CONSTANT)
  for x in range(padH, m+padH, 1): #Recorremos la distancia de la imagen
    for y in range(padV, n+padV, 1): # original dentro de la paddeada
      limar = x-center[0]
      limab = x+p-center[0]
      limizq = y-center[1]
      limder = y+q-center[1]
      window = pad[limar:limab, limizq:limder]
      outImage[x-padH,y-padV]=inImage[x-padH, y-padV]
      # Se comprueba que se dilata la región en torno al píxel
      for i in range(p):
        for j in range(q):
          if window[i][j]==1 and SE[i][j]==1:
            outImage[x-padH, y-padV]=1
  return outImage

def invert(image): # Invierte una imagen binaria en forma de ndarray
  return 1 - np.asarray(image)

########################################################

# def fillWin(window, SE, Ac, x, y):
#   m, n = window.shape
#   result = window.copy()
#   filled = set([])
#   for i in range(m):
#     for j in range(n):
#       if ((window[i][j]==0 and SE[i][j]==1) and Ac[i][j]==1):
#         filled.add((i+x-1,j+y-1))
#         result[i][j]=1
#   return result, filled

# def fill(inImage, seeds, SE=[], center=[]):
#   if SE == []:
#     # Por defecto se considera conectividad-4
#     SE = np.array([[0,1,0],[1,1,1],[0,1,0]])
#   m, n = inImage.shape
#   p, q = SE.shape
#   a, b = p//2, q//2
#   if center == []:
#     center=[a,b]
#   # outImage = np.zeros((m,n), dtype='float32')
#   outImage = inImage.copy()
#   inverted = 1-np.asarray(inImage)
#   # show(inImage, inverted)
#   filled = set([])
#   for (sx, sy) in seeds:
#     filled.add((sx,sy))
#     while len(filled)>0:
#       (sx,sy)=filled.pop()
#       limar = max(0,sx-a)
#       limab = min(m,sx+p-a)
#       limizq = max(0,sy-b)
#       limder = min(n,sy+q-b)
#       window = outImage[limar:limab, limizq:limder]
#       Ac = inverted[limar:limab, limizq:limder]
#       # for x in range(p):
#       #   for y in range(q):
#       #     if (inImage[sx+x,sy+y]==0 and inverted[sx+x,sy+y]==1):
#       #       outImage[sx+x,sy+y]=SE[x,y]
#       #       filled.add((sx+1,sy+y))
#       r, f = fillWin(window, SE, Ac, sx, sy)
#       outImage[limar:limab, limizq:limder] = r
#       filled = filled.union(f)
#   return outImage

#######

def fillWin(window, SE, Ac, x, y, toFill):
  m, n = window.shape
  result = window.copy()
  for i in range(m):
    for j in range(n):
      if ((window[i][j]==0 and SE[i][j]==1) and Ac[i][j]==1):
        # print("[LLenado el píxel (",i+x-1,",",j+y-1,")]")
        toFill.add((i+x-1,j+y-1))
        result[i][j]=1
  return result

def fill(inImage, seeds, SE=[], center=[]):
  if SE == []:
    # Por defecto se considera conectividad-4
    SE = np.array([[0,1,0],[1,1,1],[0,1,0]])
  m, n = inImage.shape
  p, q = SE.shape
  a, b = p//2, q//2
  if center == []:
    center=[a, b]
  outImage = cv.copyMakeBorder(inImage,a,a,b,b,cv.BORDER_CONSTANT, value=1)
  inverted = 1 - outImage
  # show(inImage, inverted)
  toFill = set([])
  for (sx, sy) in seeds:
    # Se ajusta el valor de la seed con el padding
    toFill.add((sx+a,sy+b))
    while len(toFill)>0:
      sx,sy=toFill.pop()
      ar, ab, iz, de = sx-center[0], sx+p-center[0], sy-center[1], sy+q-center[1]
      # Se coge siempre una ventana del tamaño del SE
      window = outImage[ar:ab, iz:de]
      Ac = inverted[ar:ab, iz:de]
      outImage[ar:ab, iz:de] = fillWin(window, SE, Ac, sx, sy, toFill)
  return outImage[a:m+a,b:n+b]

#####

def main():
  # image = read_img("./imagenes/morphology/closed.png")
  # image = read_img("./imagenes/morphology/closed44.png")
  # image = read_img("./imagenes/morphology/closed10.png")
  image = read_img("./imagenes/morphology/closed2.png")
  # image = read_img("./imagenes/morphology/closed_cut.png")

  ### Erode ###

  # SE = [[0,1,0],[1,1,1],[0,1,0]]
  # seeds = [[2,2]]
  # seeds = [[0,4]]
  seeds = [[2,2],[8,8]]
  image2 = fill(image, seeds, [], [])
  show(image, image2)

if __name__ == "__main__":
  main()

