import numpy as np
import cv2 as cv
import sys
sys.path.append('../')
import p1

def suprNoMax(mags, dirs):
  """
  Aplica supresión no máxima a los bordes de la imagen para obtener
  bordes de 1 pixel de grosor.
  """
  m,n = mags.shape
  sup = np.zeros((m,n), dtype='float32')
  for i in range(1,m-1):
    for j in range(1,n-1):
      n1, n2 = 0,0
      # Horizontal
      if (0<dirs[i,j]<22.5 or 157.5<dirs[i,j]<180):
        n1, n2 = mags[i,j-1], mags[i,j+1]
      # Vertical
      elif (67.5<dirs[i,j]<112.5):
        n1, n2 = mags[i-1,j], mags[i+1,j]
      # Diagonal Ascendente
      elif (22.5<dirs[i,j]<67.5):
        n1, n2 = mags[i+1,j+1], mags[i-1,j-1]
      # Diagonal Descendente
      elif (112.5<dirs[i,j]<157.5):
        n1, n2 = mags[i+1,j-1], mags[i-1,j+1]
      # Comprobar si es máximo
      if (mags[i,j]>n1 and mags[i,j]>n2):
        sup[i,j]=mags[i,j]
  return sup

def cornerHarris(inImage, sigmaD, sigmaI, t):
  fr, fc = p1.gradientImage(inImage,"Sobel")
  return outCorners, harrisMap


#####

def main():
  # image = p1.read_img("../imagenes/morphology/closed.png")
  # image = p1.read_img("../imagenes/grad7.png")
  image = p1.read_img("../imagenes/lena.png")

  # img = cornerHarris(image, 1.5, 0.2, 0.5)
  # p1.show(image, img)

if __name__ == "__main__":
  main()

