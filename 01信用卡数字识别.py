# 导入工具包
# contours：imutils提供的轮廓排序工具
from imutils import contours
# 数值计算库
import numpy as np
# 命令行参数解析
import argparse
# imutils 图像预处理工具（缩放、轮廓排序等）
import imutils
# OpenCV计算机视觉核心库
import cv2
# 自定义工具脚本myutils.py（resize、sort_contours函数）
import myutils

#todo:========================================固定===============================================

# 设置参数对象，用于接收命令行传入图片路径
ap = argparse.ArgumentParser()
# 添加参数-i/--image：待识别信用卡图片，必填
ap.add_argument("-i", "--image", required=True,help="path to input image")
# 添加参数-t/--template：数字模板图片（0~9标准数字模板），必填
ap.add_argument("-t", "--template", required=True,help="path to template OCR-A image")
# 解析命令行参数，转为字典args
args = vars(ap.parse_args())

# 字典：卡号第一位数字对应发卡机构
FIRST_NUMBER = {
    "3": "American Express",   # 运通
    "4": "Visa",               # Visa
    "5": "MasterCard",         # 万事达
    "6": "Discover Card"       # Discover
}

#todo:========================================固定===============================================

# 绘图展示函数：弹出窗口显示图片，按任意键关闭窗口
def cv_show(name,img):
    cv2.imshow(name, img)      # 创建窗口展示图像
    cv2.waitKey(0)             # 无限等待按键，0代表持续等待
    cv2.destroyAllWindows()    # 关闭所有opencv窗口

# todo:=====================【第一步：处理数字模板 0~9】=====================
# 读取模板图像（0-9数字标准模板图）
moban_img = cv2.imread(args["template"])
cv_show('moban_img',moban_img)

# BGR彩色图转为灰度图
moban_img0 = cv2.cvtColor(moban_img, cv2.COLOR_BGR2GRAY)
cv_show('moban_img0',moban_img0)

# 二值化：阈值10，大于10→黑色(0)，小于10→白色(255)，反向二值
# threshold返回(阈值,处理后图像)，取索引[1]得到二值图
moban_img2 = cv2.threshold(moban_img0, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('moban_img2',moban_img2)

# cv2.findContours(图片,检索模式,轮廓压缩方式)：查找图片里物体的轮廓边线
# moban_img2.copy()：传入图片副本，防止修改原图
# cv2.RETR_EXTERNAL：只找物体最外面一圈轮廓，忽略轮廓里面嵌套的小轮廓
# cv2.CHAIN_APPROX_SIMPLE：压缩轮廓点，只保留拐角点，节省内存
# 返回：轮廓列表refCnts、层级信息hierarchy
mobanCnts, hierarchy = cv2.findContours(moban_img2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# 在原图img上绘制所有轮廓，-1代表绘制全部轮廓；颜色(0,0,255)红色，线条宽度3
cv2.drawContours(moban_img,mobanCnts,-1,(0,0,255),3)
cv_show('moban_img',moban_img)
# 打印轮廓数量
print("轮廓数量：", len(mobanCnts))

# myutils自定义轮廓排序：left-to-right 从左向右排序，返回(排序后的轮廓,外接矩形)，取[0]只保留轮廓
mobanCnts = myutils.sort_contours(mobanCnts, method="left-to-right")[0]
# 创建空字典digits，用来存放：数字编号 → 数字图片
digits = {}

# for循环：enumerate同时拿到【序号i】和【轮廓c】，逐个遍历每一个数字轮廓
for (i, c) in enumerate(mobanCnts):
    # cv2.boundingRect(轮廓c)：得到包裹轮廓的最小长方形
    # 返回(x,y)长方形左上角坐标；w宽度；h高度
    (x, y, w, h) = cv2.boundingRect(c)
    # 图片切片 ref[y:y+h , x:x+w] ：截取长方形区域（把单个数字抠出来）
    roi = moban_img2[y:y + h, x:x + w]
    # cv2.resize(原图,(宽,高))：把抠出来的数字统一改成57像素宽，88像素高
    # 必须统一尺寸！后面才能和信用卡上的数字比对
    roi = cv2.resize(roi, (57, 88))
    # i就是数字0,1,2...9，保存对应的数字模板
    digits[i] = roi

# todo:=====================【第二步：处理信用卡图片】=====================
# 创建形态学操作卷积核（矩形核），大小9×3
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
# 创建5×5矩形核
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 读取待识别信用卡图片
image = cv2.imread(args["image"])
cv_show('image',image)
# 使用myutils缩放图片宽度固定为300，高度等比例缩放
image = myutils.resize(image, width=300)
# 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray',gray)

# tophat:礼帽运算：原始灰度图 - 灰度图开运算；作用：提亮图像中明亮细小区域，突出卡号
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat',tophat)

#todo:还可计算y方向水平梯度，然后两个方向做融合（这里只做了x方向）
#==================================================================
## cv2.Sobel：梯度算子，找图像明暗变化的边缘
# tophat：输入图片；ddepth=cv2.CV_32F：使用32位浮点数，防止计算信息丢失
# dx=1, dy=0：只计算水平方向边缘（卡号数字是横向摆放）
# ksize=-1：代表使用Scharr算子，检测边缘效果比普通Sobel更好
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,ksize=-1)
# 取绝对值，消除负梯度
gradX = np.absolute(gradX)
# 获取梯度图像最小值、最大值，用于归一化
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
# Min-Max归一化，映射到0~255
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
# 浮点矩阵转为uint8图像格式
gradX = gradX.astype("uint8")
#====================================================================

# 打印梯度图shape
print (np.array(gradX).shape)
cv_show('gradX',gradX)

# 闭运算：先膨胀、再腐蚀，把同一组4个数字之间缝隙填充，让一组卡号连成一个连通区域
gradX_close = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
cv_show('gradX_close ',gradX_close )

# 再次二值化；0代表让程序自动计算最佳阈值；
# cv2.THRESH_BINARY | cv2.THRESH_OTSU：自动阈值二值化模式
thresh2 = cv2.threshold(gradX_close, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh2',thresh2)

# 再次闭运算，进一步消除孔洞，强化连通区域
thresh_close = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, sqKernel)
cv_show('thresh_close',thresh_close)

# 在二值图上查找所有外轮廓(参1：复制闭运算后的thresh_close图，参2：轮廓检索模式，参3：轮廓逼近方法)
threshCnts, hierarchy = cv2.findContours(thresh_close.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# 轮廓赋值给cnts
cnts = threshCnts
# 复制原图用来绘制轮廓，避免修改原始image（信用卡图片）
cur_img = image.copy()
# 在图上画出所有找到的轮廓，红色，线宽3
cv2.drawContours(cur_img,cnts,-1,(0,0,255),3)
cv_show('img',cur_img)
# 创建空列表locs，用来保存【符合条件的4位数字组坐标】
locs = []

# 遍历全部轮廓，筛选真正卡号区域，过滤噪声杂物
for (i, c) in enumerate(cnts):
    # 获取轮廓外接矩形
    (x, y, w, h) = cv2.boundingRect(c)
    # ar = 宽 / 高，长宽比；float(h)防止整数除法出错
    ar = w / float(h)

    # 筛选条件1：4个并排数字的长方形，长宽比大约 2.5 ~4.0之间
    if ar > 2.5 and ar < 4.0:
        # 筛选条件2：限制矩形宽高范围，过滤太大、太小的干扰轮廓
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            # 满足所有条件 → 保存这个数字组坐标
            locs.append((x, y, w, h))
# 将所有数字组按照左上角x坐标从左向右排序
locs = sorted(locs, key=lambda x:x[0])
# 存放整张卡识别出来的全部卡号字符
output = []

#todo: =====================【第三步：分割每组4个数字，模板匹配识别】=====================
# 遍历每一组4位数字区域
for (i, (gX, gY, gW, gH)) in enumerate(locs):
    # 保存当前这一组识别出的4个数字
    groupOutput = []
    # 截取该组区域，上下左右向外扩充5像素（padding，防止裁剪丢失数字边缘）
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group',group)
    # OTSU自动阈值二值化，分离数字和背景
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group',group)
    # 在组图像中查找单个数字轮廓
    digitCnts,hierarchy = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # imutils.contours工具，轮廓从左向右排序，保证数字顺序正确
    digitCnts = contours.sort_contours(digitCnts,
        method="left-to-right")[0]

    # 遍历当前组里面每一个数字轮廓
    for c in digitCnts:
        # 单个数字外接矩形
        (x, y, w, h) = cv2.boundingRect(c)
        # 截取单个数字
        roi = group[y:y + h, x:x + w]
        # resize 和模板图片尺寸统一 (57,88)，才能模板匹配
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi',roi)
        # 存放和0~9每个模板的匹配分数
        scores = []
        # # 循环字典里全部0~9标准数字模板
        for (digit, digitROI) in digits.items():
            # cv2.matchTemplate(待识别图片,模板图片,匹配算法) 模板匹配
            # cv2.TM_CCOEFF：相关系数匹配算法，数值越高代表长得越像
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            # cv2.minMaxLoc：拿到匹配结果里最小值、最大值、对应坐标
            # 最大值score越高，两个图片越相似
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
        # 取最大分数对应的索引，就是识别出的数字，转字符串存入组结果
        groupOutput.append(str(np.argmax(scores)))

    # 在原图上绘制当前数字组矩形框
    cv2.rectangle(image, (gX - 5, gY - 5),
        (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    # 在矩形上方写入识别到的4位数字文本
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    # 将本组识别结果追加到全局卡号列表
    output.extend(groupOutput)

# output[0]是卡号第一位，匹配卡组织字典输出卡片类型
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
# 拼接列表字符，输出完整卡号
print("Credit Card #: {}".format("".join(output)))
# 弹出最终标记完成的效果图
cv2.imshow("Image", image)
# 等待按键
cv2.waitKey(0)