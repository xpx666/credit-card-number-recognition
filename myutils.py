# 导入opencv视觉库
import cv2

# 定义轮廓排序函数
# cnts：检测得到的轮廓列表
# method：排序方式，支持 left-to-right/right-to-left/top-to-bottom/bottom-to-top
def sort_contours(cnts, method="left-to-right"):
    # 是否反向排序标记
    reverse = False
    # 排序依据坐标：0代表x(左右)，1代表y(上下)
    i = 0

    # 如果是从右往左、从下往上排序，开启逆序
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # 如果是上下方向排序，使用y坐标作为排序标准
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # 遍历所有轮廓，得到每个轮廓的外接矩形 (x,y,w,h)
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]

    # 将轮廓和对应的外接矩形打包，按照选定坐标进行排序
    # b[1][i]：取外接矩形的x或y坐标作为排序key
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                          key=lambda b: b[1][i], reverse=reverse))
    # 返回排序后的轮廓列表、对应的外接矩形列表
    return cnts, boundingBoxes


# 自定义等比例缩放函数
# image：原始图像
# width：目标宽度；height：目标高度，只需传入其一自动等比例
# inter：插值方式，默认INTER_AREA适合缩小图像
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # 定义存放目标宽高的变量
    dim = None
    # 获取原图高度、宽度（shape[:2] 只取h,w，舍弃通道维度）
    (h, w) = image.shape[:2]

    # 宽高都不传，无需缩放，直接返回原图
    if width is None and height is None:
        return image

    # 只指定高度，按高度计算缩放比例，宽度等比例变化
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    # 只指定宽度，按宽度计算缩放比例，高度等比例变化
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # 执行图像缩放
    resized = cv2.resize(image, dim, interpolation=inter)
    # 返回缩放完成后的图像
    return resized