import numpy as np
import cv2 as cv
import sys
sys.path.append('../')
import p1

# Necesario:
# - Operación básica de llenar una región con el SE
# - Recorrer una región según una seed
# - Recorrer región si la seed no es el primer píxel de la región
# - Registrar regiones llenadas
# - Aplicar para varias seeds

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
  # image = p1.read_img("../imagenes/morphology/closed.png")
  # image = p1.read_img("../imagenes/morphology/closed44.png")
  # image = p1.read_img("../imagenes/morphology/closed10.png")
  image = p1.read_img("../imagenes/morphology/closed2.png")
  # image = p1.read_img("../imagenes/morphology/closed_cut.png")

  ### Erode ###

  # SE = [[0,1,0],[1,1,1],[0,1,0]]
  # seeds = [[2,2]]
  # seeds = [[0,4]]
  seeds = [[2,2],[8,8]]
  image2 = fill(image, seeds, [], [])
  p1.show(image, image2)

if __name__ == "__main__":
  main()

