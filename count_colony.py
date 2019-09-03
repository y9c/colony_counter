#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 yech <yech1990@gmail.com>
# Distributed under terms of the MIT license.
#
# Created: 2019-08-18 18:09

"""spot plate colony detection."""

import random

import cv2
import numpy as np
from scipy import ndimage


def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if rowmax > gray.shape[0]:
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if colmax > gray.shape[1]:
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(
        blockImage,
        (gray.shape[1], gray.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]
    ).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


def detect_spot_circle(src_img, save_img=False):
    # color to gray
    grey_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    grey_img = cv2.medianBlur(grey_img, 5)

    approx_gap = grey_img.shape[0] / 10
    # fix contract
    grey_img = adjust_gamma(grey_img, gamma=0.4)
    circles = cv2.HoughCircles(
        grey_img,
        cv2.HOUGH_GRADIENT,
        1,
        int(approx_gap * 2),
        param1=10,
        param2=20,
        minRadius=int(approx_gap / 2),
        maxRadius=int(1.2 * approx_gap),
    )
    print("#######", len(circles[0]), "spots detected!")

    circles = [
        sorted(
            circles[0],
            key=lambda x: (int(x[1] * 4 / src_img.shape[0]), x[0]),
            reverse=False,
        )
    ]

    if save_img:
        #  out_img = grey_img
        out_img = src_img.copy()
        for i, c in enumerate(circles[0], 1):
            c = c.astype(int)
            x, y, r = c
            # draw cirlce in crop img
            out_img = cv2.circle(
                out_img, (x, y), r + int(np.ceil(r / 25)), (0, 0, 255), 2
            )
            cv2.putText(
                out_img,
                str(i),
                (x + int(r * 0.85), y + int(r * 0.95)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imwrite("highlight_spots.png", out_img)
    return circles


def count_spot_colonies(circle, save_img=False):
    circle = circle.astype(int)
    x, y, r = circle
    # adjust circle
    a = int(0.08 * r)
    scale = 2
    # get the actual cropped images here
    copy_img = src_img.copy()
    crop_img = copy_img[
        max(y - r - a, 0) : min(y + r + a, copy_img.shape[0]),
        max(x - r - a, 0) : min(x + r + a, copy_img.shape[1]),
    ]

    # enlarge image
    crop_img = cv2.resize(
        crop_img, (crop_img.shape[0] * scale, crop_img.shape[1] * scale)
    )

    # fix light
    crop_img = unevenLightCompensate(crop_img, 8)

    # increase contact
    #  crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    #  crop_img = cv2.equalizeHist(crop_img)
    # increase contact (gamma correction)
    crop_img = adjust_gamma(crop_img, gamma=0.5)

    comp_img = crop_img.copy()
    #  comp_img = cv2.cvtColor(comp_img, cv2.COLOR_GRAY2BGR)

    # Convert image to gray
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    # blur image
    #  crop_img = cv2.blur(crop_img, (1, 2))

    # sobel filter
    #  crop_img = sobel_filters(crop_img)[0].astype(int)

    # detect edge
    crop_img = cv2.Canny(crop_img, 25, 75)
    # closed the edges
    #  crop_img = cv2.dilate(crop_img, None, iterations=1)
    #  crop_img = cv2.erode(crop_img, None, iterations=1)

    # RETR_TREE will count clonny twice
    contours, hierarchy = cv2.findContours(
        crop_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    # Draw contours
    cont_img = np.zeros(
        (crop_img.shape[0], crop_img.shape[1], 3), dtype=np.uint8
    )

    colony_num = 0
    for i, c in enumerate(contours):
        #  if hierarchy[0][i][2] > 0:
        #  if cv2.contourArea(c) > 5:
        #  if cv2.isContourConvex(c):
        #  color = (255, 255, 255)
        #  color = (0, 0, 0)
        #  continue

        color = (
            random.randint(0, 256),
            random.randint(0, 256),
            random.randint(0, 256),
        )
        if cv2.isContourConvex(c):
            colony_num += 1
            cv2.drawContours(
                cont_img, contours, i, color, 2, cv2.LINE_8, hierarchy, 20
            )
        else:
            #  _, _, w, h = cv2.boundingRect(c)
            _, (w, h), _ = cv2.minAreaRect(c)
            area = cv2.contourArea(c)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if (
                hull_area > 0
                and area / hull_area > 0.3
                and h < crop_img.shape[0] / 8
                and w < crop_img.shape[1] / 8
            ):
                if 1 / 1.5 < w / h < 1.5:
                    colony_num += 1
                    cv2.drawContours(
                        cont_img,
                        contours,
                        i,
                        color,
                        2,
                        cv2.LINE_8,
                        hierarchy,
                        20,
                    )
                elif 1 / 3 < w / h < 3:
                    color = (255, 255, 255)
                    cv2.drawContours(
                        cont_img,
                        contours,
                        i,
                        color,
                        1,
                        cv2.LINE_8,
                        hierarchy,
                        20,
                    )
                #  for testing and debug
                #  if save_img == "color_25.png":
                #  print("#######", w / h, color)

        # compute the Convex Hull of the contour
        #  hull = cv2.convexHull(c)
        #  cv2.drawContours(cont_img, [hull], 0, color, 2)

    # draw cirlce in crop img
    r, a = r * scale, a * scale
    cont_img = cv2.circle(
        cont_img, (r + a, r + a), r + int(np.ceil(r / 25)), (255, 255, 255), 1
    )

    out_img = np.hstack(
        (
            #  src_img.copy()[y - r - a : y + r + a, x - r - a : x + r + a],
            comp_img,
            cont_img,
        )
    )

    cv2.imwrite(save_img, out_img)
    #  cv2.imshow("detected circles", mask)
    #  cv2.waitKey(0)
    return colony_num


if __name__ == "__main__":
    # Loads an image
    #  src_img = cv2.imread("CDA_hsAID_8x5.png", cv2.IMREAD_COLOR)
    #  src_img = cv2.imread("a1-2.jpg", cv2.IMREAD_COLOR)
    src_img = cv2.imread(
        "../cda_enzyme_screen/hsAID_gWT.png", cv2.IMREAD_COLOR
    )
    # detect spot
    circles = detect_spot_circle(src_img, save_img="./highlight_spots.png")
    # count spot colony
    for index, c in enumerate(circles[0], 1):
        colony_num = count_spot_colonies(c, save_img=f"color_{index}.png")
        print(index, colony_num)
