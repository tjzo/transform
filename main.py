import math

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]


def print_marks(bg, mark_pts, radius):
    for dx in range(-radius, radius):
        for dy in range(-radius, radius):
            for id, pt in enumerate(mark_pts):
                x, y = math.floor(pt[0] + dx), math.floor(pt[1] + dy)
                if 0 <= x < bg.shape[0] and 0 <= y < bg.shape[1]:
                    bg[x][y] = color[id]


def print_bg(bg, logo, logo_offset, alpha, src_pts_2d, dst_pts_2d):
    for logo_x, l in enumerate(logo):
        for logo_y, pt in enumerate(l):
            bg_x, bg_y = logo_offset + [logo_x, logo_y]

            if 0 <= bg_x < bg.shape[0] and 0 <= bg_y < bg.shape[1]:
                for c in range(3):
                    bg[bg_x][bg_y][c] = bg[bg_x][bg_y][c] * (1 - alpha) + logo[logo_x][logo_y][c] * alpha
    print_marks(bg, src_pts_2d, 5)
    print_marks(bg, dst_pts_2d, 5)

    plt.imshow(bg)
    plt.show()


'''
calculate det of
|x y z|
|v1   |
|v2   |
result represent as a*x + b*y + c*z
return a, b, c
'''
def calc_det_3d(v1, v2):
    mat = np.array([v1, v2])
    a = np.linalg.det(mat[:, 1:])
    b = np.linalg.det(mat[:, ::2])
    c = np.linalg.det(mat[:, :2])
    return [a, -b, c]


def calc_z(pts_3d, x, y):
    v1 = pts_3d[1] - pts_3d[0]
    v2 = pts_3d[2] - pts_3d[0]
    a, b, c = calc_det_3d(v1, v2)
    if c == 0:
        # raise Exception('')
        return 0
    z = -(a * (x - pts_3d[0][0]) + b * (y - pts_3d[0][1])) / c + pts_3d[0][2]
    return z


def cvt_2d_3d(src_pts_2d, dst_pts_2d):
    src_pts_3d = np.pad(src_pts_2d, (0, 1))
    dst_pts_3d = np.pad(dst_pts_2d, (0, 1))

    for i, _ in enumerate(src_pts_2d):
        l1 = np.linalg.norm(src_pts_2d[i] - src_pts_2d[0])
        l2 = np.linalg.norm(dst_pts_2d[i] - dst_pts_2d[0])
        if l1 < l2:
            src_pts_3d[i][2] = (l2 ** 2 - l1 ** 2) ** 0.5
        else:
            dst_pts_3d[i][2] = (l1 ** 2 - l2 ** 2) ** 0.5
    return src_pts_3d, dst_pts_3d


def check_args(alpha, src_pts_2d, dst_pts_2d):
    if len(src_pts_2d) != 3:
        raise Exception('need three src points')
    if len(dst_pts_2d) != 3:
        raise Exception('need three dst points')
    for pt in src_pts_2d:
        if len(pt) != 2:
            raise Exception('not 2d points')
    for pt in dst_pts_2d:
        if len(pt) != 2:
            raise Exception('not 2d points')
    if alpha > 1 or alpha < 0:
        raise Exception('invalid alpha')


def calc_transform(src_pts_3d, dst_pts_3d, src_pt_3d):
    s1 = np.linalg.det([
        [1, 1, 1],
        src_pts_3d[1] - src_pts_3d[0],
        src_pts_3d[2] - src_pts_3d[0],
    ])

    s2 = np.linalg.det([
        [1, 1, 1],
        dst_pts_3d[1] - dst_pts_3d[0],
        dst_pts_3d[2] - dst_pts_3d[0],
    ])

    # 缩放比例
    scale = s2 / s1

    # A'B'C'P'四点共面
    v0 = calc_det_3d(dst_pts_3d[1] - dst_pts_3d[0], dst_pts_3d[2] - dst_pts_3d[0])

    # ABP A'B'P'面积等比缩放
    v1 = calc_det_3d([1, 1, 1], dst_pts_3d[1] - dst_pts_3d[0])
    r1 = np.linalg.det([
        [1, 1, 1],
        src_pts_3d[1] - src_pts_3d[0],
        src_pt_3d - src_pts_3d[0],
    ])
    # ACP A'C'P'面积等比缩放
    v2 = calc_det_3d([1, 1, 1], dst_pts_3d[2] - dst_pts_3d[0])
    r2 = np.linalg.det([
        [1, 1, 1],
        src_pts_3d[2] - src_pts_3d[0],
        src_pt_3d - src_pts_3d[0],
    ])

    # 高斯消元解方程计算P'
    res = np.linalg.solve(np.array([v0, v1, v2]), np.array([0, r1 * scale, r2 * scale]).T).T
    # print('!!', dst_pts_3d[0], res.T, (res.T)[0], dst_pts_3d[0] + res.T[0])
    dst_pt_3d = dst_pts_3d[0] + res

    return dst_pt_3d


def spread(bg, spread_list, alpha):
    spread_list.sort(key=lambda x: (x[0][0], x[0][1], x[1]))
    i = 0
    eps = 1e-10
    while i < len(spread_list):
        j = i
        x, y = spread_list[i][0]
        while j < len(spread_list) and spread_list[i][0] == spread_list[j][0]:
            j += 1
        if spread_list[i][1] < eps:
            for c in range(3):
                bg[x][y][c] = bg[x][y][c] * (1 - alpha) + spread_list[i][2][c] * alpha
        else:
            tot_weight = sum(1 / dist for _, dist, _ in spread_list[i:j])
            for c in range(3):
                bg[x][y][c] = bg[x][y][c] * (1 - alpha)
            for _, dist, colors in spread_list[i:j]:
                for c in range(3):
                    bg[x][y][c] += colors[c] * alpha / dist / tot_weight
        i = j


def print_transform(bg, logo, logo_offset, alpha, src_pts_2d, dst_pts_2d):
    check_args(alpha, src_pts_2d, dst_pts_2d)

    src_pts_3d, dst_pts_3d = cvt_2d_3d(src_pts_2d, dst_pts_2d)

    # 每个点渲染周边点的范围
    spread_radius = 1

    spread_list = []

    for logo_x, l in enumerate(logo):
        for logo_y, pt in enumerate(l):
            src_pt_2d = logo_offset + [logo_x, logo_y]
            src_pt_3d = np.array([src_pt_2d[0], src_pt_2d[1], calc_z(src_pts_3d, src_pt_2d[0], src_pt_2d[1])])

            dst_pt_3d = calc_transform(src_pts_3d, dst_pts_3d, src_pt_3d)

            bg_x, bg_y, _ = dst_pt_3d

            for dx in range(spread_radius, -spread_radius, -1):
                for dy in range(spread_radius, -spread_radius, -1):
                    tx = math.floor(bg_x + dx)
                    ty = math.floor(bg_y + dy)
                    if 0 <= tx < bg.shape[0] and 0 <= ty < bg.shape[1]:
                        dist = math.fabs((tx - bg_x) * (bg_y - ty))
                        spread_list.append(((tx, ty), dist, logo[logo_x][logo_y]))

    spread(bg, spread_list, alpha)

    plt.imshow(bg)
    plt.show()


if __name__ == '__main__':
    bg = cv.imread('test.jpg', cv.IMREAD_UNCHANGED)
    bg = cv.cvtColor(bg, cv.COLOR_BGR2RGB)
    print('size bg', bg.shape[0], bg.shape[1], bg.shape[2])

    logo = cv.imread('logo.jpg', cv.IMREAD_UNCHANGED)
    logo = cv.cvtColor(logo, cv.COLOR_BGR2RGB)
    print('size logo', logo.shape[0], logo.shape[1], logo.shape[2])
    logo_offset = np.array([100, 100])

    src_pts_2d = np.array([
        logo_offset,
        [logo_offset[0] + logo.shape[0] * 0.7, logo_offset[1]],
        [logo_offset[0] + logo.shape[0], logo_offset[1] + logo.shape[1] *0.5],
    ])
    dst_pts_2d = np.array([
        [400, 400],
        [320, 400 + logo.shape[0] * 0.75],
        [400 - logo.shape[1] * 0.8, 400 + logo.shape[0] * 0.36],
    ])
    alpha = 0.8

    print_bg(bg, logo, logo_offset, alpha, src_pts_2d, dst_pts_2d)

    print_transform(bg, logo, logo_offset, alpha, src_pts_2d, dst_pts_2d)
