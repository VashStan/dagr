import numba
import numpy as np

#增加代码
#@numba.jit(nopython=True)
def draw_events_on_image(img, x, y, p, alpha=0.5):
    if not isinstance(img, np.ndarray):
        img = np.array(img)  # 将 PIL.Image 转换为 numpy 数组

    img_copy = img.copy()
    for i in range(len(p)):
        if y[i] < len(img):
            img[y[i], x[i], :] = alpha * img_copy[y[i], x[i], :]
            img[y[i], x[i], int(p[i]) - 1] += 255 * (1 - alpha)
    return img