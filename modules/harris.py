import numpy as np
import cv2 as cv
import sys
sys.path.append('../')
import p1

def corners(harrisMap):


def cornerHarris(inImage, sigmaD, sigmaI, t):
  # m, n = inImage.shape
  ix, iy = p1.gradientImage(inImage,"Sobel")
  # show(adjustIntensity(ix,[],[0,1]),adjustIntensity(iy,[],[0,1]))
  ix2, iy2, ixy = ix**2, iy**2, ix*iy
  # show(ix2, iy2)
  # show1(adjustIntensity(ixy,[],[0,1]))
  gix2 = p1.gaussianFilter(ix2,sigmaD)
  giy2 = p1.gaussianFilter(iy2,sigmaD)
  gixiy = p1.gaussianFilter(ixy,sigmaD)
  k = 0.05
  detA = gix2*giy2 - gixiy**2 # aplicar gaussiana con sigmaI(?)
  trace = gix2+giy2
  response = p1.gaussianFilter(detA,sigmaI) - k*trace**2
  harrisMap = p1.adjustIntensity(response,[],[0,1])
  sup = p1.suprNoMax(np.sqrt((ix**2)+(iy**2)),p1.direcciones(ix,iy))
  # outCorners = p1.umbr(1-harrisMap, t)
  outCorners = p1.umbr(sup, t)
  # return outCorners, 1-harrisMap #visualización igual que en los ejemplos de clase
  return outCorners, harrisMap


#####

def main():
  image = p1.read_img("../imagenes/grid.png")

  cornersHarris, harrisMap = cornerHarris(image, 1.5, 0.5, 0.4)
  # p1.show(image, harrisMap)
  p1.show(image, cornerHarris)

if __name__ == "__main__":
  main()

