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
      # Comprobar si es máximo
      if (mags[i,j]>n1 and mags[i,j]>n2):
        sup[i,j]=mags[i,j]
  return sup

def umbralizacionConHisteresis(sup, tlow, thigh):
  """
  Se obtiene una imagen umbralizada a partir del resultado de aplicar
  la supresión no máxima.
  Se aunan todos los bordes, juntando los débiles con los fuertes si están 4-conectados.
  """
  m, n = sup.shape
  # Imagen con padding para no coger puntos fuera de la imagen
  supAux = cv.copyMakeBorder(sup,1,1,1,1,cv.BORDER_CONSTANT,value=0.0)
  umbr = np.zeros((m,n), dtype='float32')
  for i in range(1,m,1):
    for j in range(1,n,1):
      # Se comprueba si el valor sobrepasa el umbral
      if (supAux[i,j]>thigh):
        umbr[i-1,j-1]=1.0
      # Sino, se comprueba si está 4-conectado a un valor que sobrepase thigh
      elif (supAux[i,j]>tlow) and ((supAux[i+1,j]+supAux[i-1,j]+supAux[i,j+1]+supAux[i,j-1])>0):
        umbr[i-1,j-1]=1.0
  return umbr


def edgeCanny(inImage, sigma, tlow, thigh):
  direcciones, magnitudes = mejora(inImage,sigma)
  supresion = suprNoMax(magnitudes, direcciones)
  umbralizado = umbralizacionConHisteresis(supresion,tlow, thigh)
  return umbralizado


#####

def main():
  # image = p1.read_img("../imagenes/morphology/closed.png")
  # image = p1.read_img("../imagenes/grad7.png")
  image = p1.read_img("../imagenes/lena.png")

  img = edgeCanny(image, 1.5, 0.2, 0.5)
  p1.show(image, img)

if __name__ == "__main__":
  main()

