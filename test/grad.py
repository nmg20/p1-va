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

def adjustIntensity(inImage, inRange=[], outRange=[0, 1]): # [2]
  """
  Altera el rango dinámico de la imagen.
  - inRange = Rango de valores de intensidad de entrada.
  - outRange = Rango de valores de intensidad de salida.
  """
  if inRange == []:
    min_in = np.min(inImage)
    max_in = np.max(inImage)
  else :
    min_in = inRange[0]
    max_in = inRange[1]
  min_out = outRange[0]
  max_out = outRange[1]
  # print("MinIn: ",min_in,"\tMaxIn: ",max_in,"\n\tMinOut: ",min_out,"\tMaxOut: ",max_out)
  return min_out + (((max_out - min_out)*inImage - min_in)/(max_in - min_in))



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
  a = np.array([[-1,0,1]])
  b = np.array([[1,1,1]])
  c = np.array([[1,2,1]])
  if operator == "roberts":
    mx, my = np.array([[-1,0],[0,1]]), np.array([[0,-1],[1,0]])
  elif operator == "prewitt":
    mx, my = b.T * a, a.T * b
  elif operator == "sobel":
    mx, my = c.T * a, a.T*c
  p, q = mx.shape # Se toman las dimensiones de una matriz indistinta de las dos para hacer padding
  a, b = p//2, q//2
  pad = cv.copyMakeBorder(inImage,a,a,b,b,cv.BORDER_CONSTANT)
  for x in range(a,m+a,1):
    for y in range(b,n+b,1):
      # Cogemos la ventana de la imagen paddeada
      window = pad[(x-a):(x+p-a),(y-b):(y+q-b)]
      gx[x-a,y-b] = (window*mx).sum() # Cambiar por vectores para hacer más eficiente
      gy[x-a,y-b] = (window*my).sum() # -> matrices linealmente separables
  return [gx, gy]

#####

def main():
  # image = read_img("./imagenes/morphology/closed.png")
  # image = read_img("./imagenes/grad7.png")
  image = read_img("./imagenes/lena.png")

  ### Erode ###

  # SE = [[0,1,0],[1,1,1],[0,1,0]]
  # seeds = [[1,1],[4,2]]
  gx, gy = gradientImage(image, "roberts")
  gx = adjustIntensity(gx, [], [0,1])
  gy = adjustIntensity(gy, [], [0,1])
  show(image, gx)
  show(image, gy)

if __name__ == "__main__":
  main()

