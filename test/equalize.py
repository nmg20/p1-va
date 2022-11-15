import numpy as np
import cv2 as cv
import math
import imutils  #Sólo se usa en show2 para poder hacer resize proporcional de las imágenes
import matplotlib.pyplot as plt
import p1

def plotHist(inImage, outImage):
  ax1 = plt.subplot(221)
  ax1.imshow(inImage, cmap='gray', vmax=1, vmin=0)
  ax1.set_title("Original")
  ax1.set_axis_off()

  ax2 = plt.subplot(222, sharex=ax1, sharey=ax1)
  ax2.imshow(outImage, cmap='gray', vmax=1, vmin=0)
  ax2.set_title("Modificada")
  ax2.set_axis_off()

  ax3 = plt.subplot(223)
  ax3.hist(inImage.ravel() * 255, bins=256, range=(0,255))
  ax3.set_xlim(0, 255)
  ax3.autoscale(enable=True, axis='y', tight=True)
  ax3.set_xlabel("Intensidad")
  ax3.set_ylabel("Frecuencia")

  ax4 = plt.subplot(224, sharex=ax3, sharey=ax3)
  ax4.hist(outImage.ravel() * 255, bins=256, range=(0,255))
  ax4.set_xlim(0, 255)
  ax4.set_xlabel("Intensidad")
  ax4.set_ylabel("Frecuencia")

  plt.tight_layout()
  plt.show()


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


# def getHist(image, bins):
#   hist = np.zeros(bins, dtype='int')
#   binsAux = np.linspace(np.min(image), np.max(image), bins)
#   for x in np.nditer(img):
#     i = 0
#     while x>binsAux[i]:
#       i=i+1
#     hist[i-1] = hist[i-1]+1
#   return hist

# def cumsum(hist):
#   l = [next(hist)]
#   for i in np.nditer(hist):
#     l.append(l[-1]+i)
#   return np.array(l)

# def showHist(img):
#   plt.hist()

#############################

def equalizeIntensity(inImage, nBins=256):
  hist = np.zeros(nBins, dtype='int')
  binsAux = np.linspace(np.min(inImage), np.max(inImage), nBins+1)
  outImage = np.zeros((np.shape(inImage)),dtype='float32')
  for x in np.nditer(inImage):
    i = 0
    while x>binsAux[i]:
      i=i+1
    hist[i-1] = hist[i-1]+1
  cdf = hist.cumsum()
  c_norm = cdf * hist.max() / cdf.max()
  outImage = np.interp(inImage, binsAux[:-1], c_norm)
  outImage = adjustIntensity(outImage,[],[0,1])
  return outImage

#####

def main():
  # image = p1.read_img("./imagenes/circles.png")
  # image = p1.read_img("./imagenes/circles1.png")
  image = p1.read_img("./imagenes/eq.jpg")
  image2 = equalizeIntensity(image, 256)
  plotHist(image, image2)

if __name__ == "__main__":
  main()

