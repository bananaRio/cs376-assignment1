import numpy as np
import matplotlib.pyplot as plt
import cv2

def plotRGB(image):
  R = image[:,:,0]
  G = image[:,:,1]
  B = image[:,:,2]
  return R, G, B

def convertToLAB(image):
  lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
  return lab_image

def plotLAB(image):
  L = image[:,:,0]
  A = image[:,:,1]
  B = image[:,:,2]
  return L, A, B

# opencv reads images in BGR format by default
# cv2.cvtColor is used to convert BGR to RGB
indoor_image = cv2.cvtColor(cv2.imread('indoor.png'), cv2.COLOR_BGR2RGB)
outdoor_image = cv2.cvtColor(cv2.imread('outdoor.png'), cv2.COLOR_BGR2RGB)

print("indoor_image (shape) ", indoor_image.shape)
print("outdoor_image (shape) ", outdoor_image.shape)

if len(indoor_image.shape) != 3 or indoor_image.shape[2] != 3:
  raise ValueError("indoor Input image must be H x W x 3")
if len(outdoor_image.shape) != 3 or outdoor_image.shape[2] != 3:
  raise ValueError("outdoor Input image must be H x W x 3")

in_R, in_G, in_B = plotRGB(indoor_image)
out_R, out_G, out_B = plotRGB(outdoor_image)
print("Indoor Image RGB Channels:")
plt.imshow(in_R, cmap='gray'); plt.title('Indoor RGB-R'); plt.savefig('indoor_RGB-R.png'); plt.clf()
plt.imshow(in_G, cmap='gray'); plt.title('Indoor RGB-G'); plt.savefig('indoor_RGB-G.png'); plt.clf()
plt.imshow(in_B, cmap='gray'); plt.title('Indoor RGB-B'); plt.savefig('indoor_RGB-B.png'); plt.clf()
print("-----")
print("Outdoor Image RGB Channels:")
plt.imshow(out_R, cmap='gray'); plt.title('Outdoor RGB-R'); plt.savefig('outdoor_RGB-R.png'); plt.clf()
plt.imshow(out_G, cmap='gray'); plt.title('Outdoor RGB-G'); plt.savefig('outdoor_RGB-G.png'); plt.clf()
plt.imshow(out_B, cmap='gray'); plt.title('Outdoor RGB-B'); plt.savefig('outdoor_RGB-B.png'); plt.clf()
print("-----")

indoor_lab = convertToLAB(indoor_image.copy())
outdoor_lab = convertToLAB(outdoor_image.copy())

in_L, in_A, in_B = plotLAB(indoor_lab)
out_L, out_A, out_B = plotLAB(outdoor_lab)

print("Indoor Image LAB Channels:")
plt.imshow(in_L, cmap='gray'); plt.title('Indoor LAB-L'); plt.savefig('indoor_LAB-L.png'); plt.clf()
plt.imshow(in_A, cmap='gray'); plt.title('Indoor LAB-A'); plt.savefig('indoor_LAB-A.png'); plt.clf()
plt.imshow(in_B, cmap='gray'); plt.title('Indoor LAB-B'); plt.savefig('indoor_LAB-B.png'); plt.clf()
print("-----")
print("Outdoor Image LAB Channels:")
plt.imshow(out_L, cmap='gray'); plt.title('Outdoor LAB-L'); plt.savefig('outdoor_LAB-L.png'); plt.clf()
plt.imshow(out_A, cmap='gray'); plt.title('Outdoor LAB-A'); plt.savefig('outdoor_LAB-A.png'); plt.clf()
plt.imshow(out_B, cmap='gray'); plt.title('Outdoor LAB-B'); plt.savefig('outdoor_LAB-B.png'); plt.clf()
print("-----")
