import matplotlib.pyplot as plt
import cv2
import numpy as np

def resizeTo256(image):
  h, w = image.shape[:2]
  # this scales to smaller side to 256
  scale = max(256/h, 256/w)
  resized = cv2.resize(image, (int(scale*w), int(scale*h)))
  # center crop to 256x256
  rh, rw = resized.shape[:2]
  y = (rh - 256) // 2
  x = (rw - 256) // 2
  return resized[y:y+256, x:x+256]

# image1 = cv2.cvtColor(cv2.imread('imgfor_cs376asn1.jpg'), cv2.COLOR_BGR2RGB)
# image2 = cv2.cvtColor(cv2.imread('imgForcs376asn1-2.jpg'), cv2.COLOR_BGR2RGB)
image1 = resizeTo256(cv2.imread('imgfor_cs376asn1.jpg'))
image2 = resizeTo256(cv2.imread('imgForcs376asn1-2.jpg'))
if cv2.imwrite('im1.jpg', image1):
  print("Saved im1.jpg")
if cv2.imwrite('im2.jpg', image2):
  print("Saved im2.jpg")

print("image1 shape: ", image1.shape)
print("image2 shape: ", image2.shape)
#plt.imshow(image1); plt.title('Image 1'); plt.savefig('im1.png'); plt.clf()

im1_lab = cv2.cvtColor(image1,cv2.COLOR_RGB2LAB).astype(float)
im2_lab = cv2.cvtColor(image2,cv2.COLOR_RGB2LAB).astype(float)

diff_L = np.sqrt((im1_lab[:,:,1]-im2_lab[:,:,1])**2 + (im1_lab[:,:,2]-im2_lab[:,:,2])**2)

# 16 px margin = 32 x 32 patch
margin = 16
# rows start at margin to end-margin
best = diff_L[margin:-margin, margin:-margin]
# here using argmax to find the largest difference
idx = np.unravel_index(np.argmax(best,axis=None), best.shape)
best_y, best_x = idx[0] + margin, idx[1] + margin
print(f"Best coord: x1={best_x}, y1={best_y}, x2={best_x}, y2={best_y}, AB dist={diff_L[best_y, best_x]:.2f}")
# visualize the best patches
patch1 = image1[best_y - margin: best_y + margin, best_x - margin: best_x + margin, :]
patch2 = image2[best_y - margin: best_y + margin, best_x - margin: best_x + margin, :]
plt.imshow(patch1); plt.title('Best Patch from Image 1'); plt.savefig('best_patch_im1.png'); plt.clf()
plt.imshow(patch2); plt.title('Best Patch from Image 2'); plt.savefig('best_patch_im2.png'); plt.clf()