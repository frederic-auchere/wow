import os
import cv2
import numpy as np


def cross_correlate(image, reference):
    cc = cv2.matchTemplate(reference, np.copy(image[512:512+1024, 512:512+1024]), cv2.TM_CCORR_NORMED)
    cy, cx = np.unravel_index(np.argmax(cc, axis=None), cc.shape)

    xi = [cx - 1, cx, cx + 1]
    yi = [cy - 1, cy, cy + 1]
    ccx2 = cc[[cy, cy, cy], xi] ** 2
    ccy2 = cc[yi, [cx, cx, cx]] ** 2

    xn = ccx2[2] - ccx2[1]
    xd = ccx2[0] - 2 * ccx2[1] + ccx2[2]
    yn = ccy2[2] - ccy2[1]
    yd = ccy2[0] - 2 * ccy2[1] + ccy2[2]

    if xd != 0:
        dx = xi[2] - (xn / xd + 0.5)
    else:
        dx = cx
    if yd != 0:
        dy = yi[2] - (yn / yd + 0.5)
    else:
        dy = cy

    dx -= 512
    dy -= 512

    return dx, dy
