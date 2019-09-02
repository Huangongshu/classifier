import cv2
import numpy as np


def square_rotation(image):
    random_angle = np.random.randint(0, 359)
    d = image.shape[1]
    r = d / 2
    m = cv2.getRotationMatrix2D(center=(r, r), angle=random_angle, scale=1)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])
    # compute the new bounding dimensions of the image
    nd = int((d * sin) + (d * cos))
    # adjust the rotation matrix to take into account translation
    m[0, 2] += (nd / 2) - r
    m[1, 2] += (nd / 2) - r
    rotated_image = cv2.warpAffine(image, m, dsize=(nd, nd), borderValue=0)
    left = int(nd / 2 - r)
    right = int(nd / 2 + r)
    top = int(nd / 2 - r)
    bottom = int(nd / 2 + r)
    return rotated_image[left:(right+1), top:(bottom+1)]


def extract_retina(image):
    x = image[:,:,:].sum(2)
    mask = x > x.mean() / 5
    kernel_size = 30
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = (mask * 255).astype(np.uint8)
    image_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    image_open = cv2.morphologyEx(image_close, cv2.MORPH_OPEN, kernel).astype(np.bool)
    image = (image * np.expand_dims(image_open, 2)).astype(np.uint8)
    retina_hrange, = np.where(image_open.sum(0) > 0)
    retina_vrange, = np.where(image_open.sum(1) > 0)
    retina_vcenter = np.argmax(image_open.sum(1))
    retina_hcenter = np.argmax(image_open.sum(0))
    left = retina_hrange.min()
    right = retina_hrange.max() + 1
    top = retina_vrange.min()
    bottom = retina_vrange.max() + 1
    width = right - left
    height = bottom - top
    if width < height:
        d = height
    else:
        d = width
    new_top = np.maximum(np.round((d / 2) - retina_vcenter).astype(np.int32), 0)
    new_left = np.maximum(np.round((d / 2) - retina_hcenter).astype(np.int32), 0)
    new_bottom = np.minimum(new_top + height, d)
    new_right = np.minimum(new_left + width, d)
    new_top = new_bottom - height
    new_left = new_right - width
    fill_image = image[top:bottom, left:right, :]
    extracted_img = np.zeros([d, d, 3], dtype=np.uint8)
    extracted_img[new_top:new_bottom, new_left:new_right, :] = fill_image
    return extracted_img


def warwick_method(image):
    scale = 500
    r = image.shape[1] / 2
    s = scale * 1.0 / r
    a = cv2.resize(image, (0, 0), fx=s, fy=s)
    black_space = a[:, :, :].sum(2) == 0
    # subtract local mean color
    a = cv2.addWeighted(a, 4, cv2.GaussianBlur(a, (0, 0), scale / 30), -4, 128)
    # remove outer 7% (adjusted)
    keep_rate = 0.93
    circle_mask = np.zeros(a.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (int(a.shape[1] / 2), int(a.shape[0] / 2)), int(scale * keep_rate), (1, 1, 1), -1, 8, 0)
    gray_img = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    brightness_mask = np.bitwise_not(np.bitwise_or(gray_img > 247, black_space))
    kernel_size = 30
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    brightness_mask_close = cv2.morphologyEx((brightness_mask * 255).astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    brightness_mask_open = cv2.morphologyEx(brightness_mask_close, cv2.MORPH_OPEN, kernel).astype(np.bool)
    mask_merged = np.bitwise_and(np.expand_dims(brightness_mask_open, 2), circle_mask)
    a = a * mask_merged + 128 * (1 - mask_merged)
    d = a.shape[1]
    remove_rate_side = (1 - keep_rate) / 2
    left = np.round(d * remove_rate_side).astype(np.int32)
    right = np.round(d * (keep_rate + remove_rate_side)).astype(np.int32)
    top = left
    bottom = right
    return a[left:right, top:bottom]


def preprocessor(image, is_training):
    image = extract_retina(image)
    if is_training:
        image = square_rotation(image)
    image = warwick_method(image)
    return image

def im_preprocess(image): #没加亮度进去，担心亮度会影响清晰度
    image=extract_retina(image)
    image=square_rotation(image)
    return image
    
if __name__=='__main__':
    image = cv2.imread("C:/Users/27011/Desktop/180101215098_20180101_143314_Color_R_001.jpg")
    image = preprocessor(image, True)
    cv2.imshow("preprocess_im", cv2.resize(image, (image.shape[0]//10,image.shape[1]//10)))
    cv2.waitKey(1000) 
    cv2.destroyAllWindows() 
 
