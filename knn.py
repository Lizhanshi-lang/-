"""
本代码通过文章《基于 ＲeliefF 特征加权和 KNN 的自然图像分类方法》的内容进行复现
===================================================================

班级：信安171
作者：葛启丰
学号：41724180
时间：2020-04-08
"""

# %%
import os
import random
import json
import cv2
import numpy as np
import array

# %%
train_data_path = os.path.join(".", "Corel-1000", "train")
test_data_path = os.path.join(".", "Corel-1000", "test")
_, categories, _ = next(os.walk(train_data_path))


# %%
# 提取特征
# 总计25个特征


# %%
def get_GLCM(img, direction):
    """
    求灰度共生矩阵

    :param img: img, BGR or GRAY
    :param direction: 0, 45, 90, 135,
    :return: 共生矩阵，大小为（256,256）
    """
    GLCM = np.zeros((256, 256), np.uint16)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows, columns = img.shape
    else:
        rows, columns = img.shape

    if direction == 0:
        for i in range(rows):
            for j in range(columns):
                if j + 1 < columns:
                    GLCM[img[i][j]][img[i][j + 1]] += 1
    elif direction == 45:
        for i in range(rows):
            for j in range(columns):
                if j + 1 < columns and i - 1 >= 0:
                    GLCM[img[i][j]][img[i - 1][j + 1]] += 1
    elif direction == 90:
        for i in range(rows):
            for j in range(columns):
                if i - 1 >= 0:
                    GLCM[img[i][j]][img[i - 1][j]] += 1
    elif direction == 135:
        for i in range(rows):
            for j in range(columns):
                if i - 1 >= 0 and j - 1 >= 0:
                    GLCM[img[i][j]][img[i - 1][j - 1]] += 1
    else:
        raise Exception("请检查输入的角度是否正确！！！")
    return GLCM


# %%
def get_energy_by_GLCM(GLCM):
    """
    根据灰度共生矩阵（GLCM）获取【能量】

    :param GLCM:
    :return:
    """
    return np.sum(GLCM ** 2)


# %%
def get_idm_by_GLCM(GLCM):
    """
    根据灰度共生矩阵（GLCM）获取【相关性】

    inverse_different_moment

    :param GLCM:
    :return:
    """
    rows, columns = GLCM.shape
    idm = 0
    for i in range(rows):
        for j in range(columns):
            idm += GLCM[i][j] / (1 + (i - j) ** 2)
    return idm


# %%
def get_sma_by_GLCM(GLCM):
    """
    根据灰度共生矩阵获取【惯性矩】
    Second moment of area: 二次统计量

    :return:
    """
    rows, columns = GLCM.shape
    sma = 0
    for i in range(rows):
        for j in range(columns):
            sma += ((i - j) ** 2) * GLCM[i][j]
    return sma


# %%
def get_rel_by_GLCM(GLCM):
    """
    根据灰度工程矩阵获取【相关性】

    """
    rows, columns = GLCM.shape
    rel = 0
    rel_x = 0
    rel_y = 0
    delta_x = 0
    delta_y = 0
    for i in range(rows):
        for j in range(columns):
            rel_x += i * GLCM[i][j]
            rel_y += j * GLCM[i][j]
    for i in range(rows):
        for j in range(columns):
            delta_x += ((i - rel_x) ** 2) * GLCM[i][j]
            delta_y += ((i - rel_y) ** 2) * GLCM[i][j]
    rel = (-rel_x * rel_y) / (delta_x * delta_y)
    return rel


# %%
def get_features(img):
    """
    输入图像，然后获取其特征向量

    BGR 各三个
    - 颜色一阶矩平均值( Average)
    - 颜色二阶矩方差( Variance)
    - 颜色三阶矩偏斜度( Skewness)

    0，π/4，π/2，3π/4 方向上的 4 个特征参数
    - 惯性矩
    - 相关性
    - 能量
    - 均匀性

    共计 3*3+4*4=25 个特征向量

    :param img: 输入图像
    :return: 返回特征向量，维数25
    """

    fea = np.zeros((25))

    img = np.array(img)

    def get_0_8(raster):
        f0 = raster.mean()
        f1 = raster.std() ** 2
        f2 = np.sum(((raster.reshape(-1) - raster.mean()) ** 3) / (raster.std() ** 3)) / (raster.size - 1)
        return f0, f1, f2

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    fea[0], fea[1], fea[2] = get_0_8(B)
    fea[3], fea[4], fea[5] = get_0_8(G)
    fea[6], fea[7], fea[8] = get_0_8(R)

    # f0 = B.mean()
    # f1 = B.std() ** 2
    # f2 = np.sum(((B.reshape(-1) - B.mean()) ** 3) / (B.std() ** 3)) / (B.size - 1)
    #
    # f3 = G.mean()
    # f4 = G.std() ** 2
    # f5 = np.sum(((G.reshape(-1) - G.mean()) ** 3) / (G.std() ** 3)) / (G.size - 1)
    #
    # f6 = R.mean()
    # f7 = R.std() ** 2
    # f8 = np.sum(((R.reshape(-1) - R.mean()) ** 3) / (R.std() ** 3)) / (R.size - 1)

    # print(f1, f2, f3)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_GLCM = get_GLCM(img_gray, 0)
    # img_GLCM = (img_GLCM - img_GLCM.min()) / (img_GLCM.max() - img_GLCM.min())

    def get_9_24(img_gray, direction):
        img_GLCM = get_GLCM(img_gray, direction)
        img_GLCM = (img_GLCM - img_GLCM.min()) / (img_GLCM.max() - img_GLCM.min())
        f0 = get_sma_by_GLCM(img_GLCM)
        f1 = get_rel_by_GLCM(img_GLCM)
        f2 = get_energy_by_GLCM(img_GLCM)
        f3 = get_idm_by_GLCM(img_GLCM)

        return f0, f1, f2, f3

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fea[9], fea[10], fea[11], fea[12] = get_9_24(img_gray, 0)
    fea[13], fea[14], fea[15], fea[16] = get_9_24(img_gray, 45)
    fea[17], fea[18], fea[19], fea[20] = get_9_24(img_gray, 90)
    fea[21], fea[22], fea[23], fea[24] = get_9_24(img_gray, 135)

    return fea


# %%
train_data_features = {}
for category in categories:
    tmp_path = os.path.join(train_data_path, category)
    _, _, imgs = next(os.walk(tmp_path))
    cur_img_fea = []
    for img in imgs:
        print(img)
        targets = os.path.join(tmp_path, img)
        img = cv2.imread(targets)
        cur_img_fea.append(get_features(img))
    train_data_features[category] = cur_img_fea.tolist()
json.dump(train_data_features, "./train_data_features")


# %%
def main():
    pass


if __name__ == "__main__":
    main()
