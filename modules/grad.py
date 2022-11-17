import numpy as np
import cv2 as cv
import sys
sys.path.append('../')
import p1

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
  d = np.array([[0,1,0]])
  if operator == "Roberts":
    mx, my = np.array([[-1,0],[0,1]]), np.array([[0,-1],[1,0]])
  elif operator == "CentralDiff":
    mx, my = a.T*d, d.T*a # Deberían ser el vector a y su transpuesta (no funciona x alguna razon)
  elif operator == "Prewitt":
    mx, my = b.T * a, a.T * b
  elif operator == "sSobel":
    mx, my = c.T * a, a.T*c
  # p, q = mx.shape # Se toman las dimensiones de una matriz indistinta de las dos para hacer padding
  # a, b = p//2, q//2
  # pad = cv.copyMakeBorder(inImage,a,a,b,b,cv.BORDER_CONSTANT)
  # for x in range(a,m+a,1):
  #   for y in range(b,n+b,1):
  #     # Cogemos la ventana de la imagen paddeada
  #     window = pad[(x-a):(x+p-a),(y-b):(y+q-b)]
  #     gx[x-a,y-b] = (window*mx).sum() # Cambiar por vectores para hacer más eficiente
  #     gy[x-a,y-b] = (window*my).sum() # -> matrices linealmente separable
  return [p1.filterImage(inImage, mx), p1.filterImage(inImage, my)]

#####

def main():
  # image = p1.read_img("../imagenes/morphology/closed.png")
  # image = p1.read_img("../imagenes/grad7.png")
  image = p1.read_img("../imagenes/lena.png")

  gx, gy = gradientImage(image, "CentralDiff")
  gx = p1.adjustIntensity(gx, [], [0,1])
  gy = p1.adjustIntensity(gy, [], [0,1])
  p1.show(image, gx)
  p1.show(image, gy)

if __name__ == "__main__":
  main()

