import numpy as np
import cv2 as cv
import sys
sys.path.append('../')
import p1

def direcciones(gx, gy):
  m,n = gx.shape
  d = np.zeros((m,n), dtype='float32')
  for i in range(m):
    for j in range(n):
      d[i,j]=np.arctan2(gy[i,j],gx[i,j])*180/np.pi
      if d[i,j]<0:
        d[i,j]=d[i,j]+180
  return d

def mejora(img, sigma):
  """
  Devuelve una matriz con las direcciones y otra con las magnitudes.
  """
  suavizado = p1.gaussianFilter(img, sigma)
  gx, gy = p1.gradientImage(suavizado, "Sobel")
  gtotal = np.sqrt((gx**2)+(gy**2))
  em = direcciones(gx, gy)
  return em, gtotal

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
      if (mags[i,j]>n1 and mags[i,j]>n2):
        sup[i,j]=mags[i,j]
  return sup

def umbralizacion(sup, tlow, thigh)
  

def histeresis()


def edgeCanny(inImage, sigma, tlow, thigh):


  return null


#####

def main():
  # image = p1.read_img("../imagenes/morphology/closed.png")
  # image = p1.read_img("../imagenes/grad7.png")
  image = p1.read_img("../imagenes/lena.png")

  # gx, gy = p1.gradientImage(image, "CentralDiff")
  # gx = p1.adjustIntensity(gx, [], [0,1])
  # gy = p1.adjustIntensity(gy, [], [0,1])
  # p1.show(image, gx)
  # p1.show(image, gy)

if __name__ == "__main__":
  main()

