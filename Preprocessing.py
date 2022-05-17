import cv2
import numpy as np
from tqdm import tqdm
#2018.10.31更新基于海天线检测
#vector<Rect> tar_detect(Mat &img)

def tar_detect(img,return_horizon=False):
	tar_pos=[]
	if img is None:
		return tar_pos

	imgray=cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
	imgray=cv2.GaussianBlur(imgray,  (3, 3), 0, 0)
	thesd = 0.0
	Mkernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
	thesd, imggtem = cv2.threshold(imgray, thesd, 255, cv2.THRESH_OTSU)
	dilatedgray = cv2.dilate(imggtem, kernel=Mkernel)

	# imgsrc = img.copy()
	# cv2.imshow('dt', dilatedgray)
	# cv2.imshow('gt', imggtem)
	# cv2.waitKey(1)

	imgmask=cv2.reduce(dilatedgray,  1, cv2.REDUCE_AVG, cv2.CV_16S)#//reduce to single column average
	row,col=imgray.shape
	#row = imgray.rows, col = imgray.cols;
	kuan_hight = round(row / 20)

	horizon_top = 0
	horizon_bottom = row - 1#;//区域上下界
	thesd,imgtem = cv2.threshold(imgmask,  thesd, 255, cv2.THRESH_OTSU)
	imgtemd=np.abs(cv2.filter2D(imgtem,cv2.CV_16S,np.array([[-1],[0],[1]])))

	bottom_temp = row - 1 #;//区域下界
	flagContinue=False
	for i in range(kuan_hight,row-1):#(int i = kuan_hight; i < row; i++)//获得海天线上下界
		ppre = imgtemd[i,0]
		#paft=imgtem[i+1,0]

		if ppre == 0:# 寻找跳变，先验认为天空255,当没有河流则可能被认为255，增加0->255的判断,抓住第一个跳变
			continue
		top_temp = i - 1.5*kuan_hight#  //海天线上界
		horizon_top = 0 if top_temp < 0 else top_temp
		bottom_temp = i + 1.5*kuan_hight
		horizon_bottom = row - 1 if bottom_temp >= row else bottom_temp #海天线下界
		break

	# horizonLine=round((horizon_bottom + horizon_top) / 2)
	# imgsrc=cv2.line(imgsrc,(0,horizonLine),(img.shape[1]-1,horizonLine),(0,0,255),2)
	# cv2.imshow('line',imgsrc)
	# cv2.waitKey(1)
	#//检查海天线区域
	if return_horizon:
		return (horizon_bottom+horizon_top)/2

	if bottom_temp-horizon_top < kuan_hight:	#return tar_posprint(tar_pos)
		return tar_pos

	imgrect=imgray[round(horizon_top):round(horizon_bottom),:]#截取海天线区域
	imgharris=cv2.cornerHarris(imgrect,  3, 3, 0.22)
	# minStrength=0.0
	# maxStrength=0.0
	#//计算最大最小响应值
	minStrength, maxStrength, minLoc, maxLoc=cv2.minMaxLoc(imgharris)#, &minStrength, &minStrength);


	localMax=None
	#//默认3*3核膨胀，膨胀之后，除了局部最大值点和原来相同，其它非局部最大值点被
	#//3*3邻域内的最大值点取代
	dilated=cv2.dilate(imgharris,kernel=None)
	#//与原图相比，只剩下和原图值相同的点，这些点都是局部最大值点，保存到localMax
	localMax=cv2.compare(imgharris, dilated,  cv2.CMP_EQ)

	cornerMap=None
	cornerTh=None
	thresholdvalue=0.0
	#// 根据角点响应最大值计算阈值
	thresholdvalue = 0.01*maxStrength
	_,cornerTh=cv2.threshold(imgharris, thresholdvalue, 255, cv2.THRESH_BINARY)
	#// 转为8-bit图
	cornerMap=cornerTh.astype(np.uint8)
	#// 和局部最大值图与，剩下角点局部最大值图，即：完成非最大值抑制
	cornerMap=cv2.bitwise_and(cornerMap, localMax)
	pts=[]#vector<Point>
	for y in range(0,cornerMap.shape[0]):#= 0; y < cornerMap.rows; y++)
		rowPtr=cornerMap[y,:]
		#const uchar* rowPtr = cornerMap.ptr<uchar>(y);
		for x in range(0,cornerMap.shape[1]):#(int x = 0; x < cornerMap.cols; x++)
			#// 非零点就是角点
			if rowPtr[x]:
				pts.append((x, y))

	tar_pos,flag_t = detect_Target(pts)
	for i in range(0,flag_t):# = 0; i < flag_t; i++)
		tar_pos[i][1] += horizon_top
	#print(tar_pos)
	return tar_pos

def detect_Target(points):#const vector<Point> &points, int *flag_target
	flag_target=0
	#rect=[0, 0, 0, 0]#Rect (xmin,ymin,w,h)
	copoints = points.copy()#vector<Point>
	target_pos=[]#vector<Rect>

	if len(copoints) >= 4:

		boat_num = 0

		r = 1100#;//4000
		while len(copoints)>0:
			p_num = 1
			#tar_top=0
			#tar_bottom=0
			#tar_left=0
			#tar_right=0
			boat_num+=1
			temp_x = copoints[0][0]
			temp_y = copoints[0][1]#;//聚类中心点
			tar_top = tar_bottom = temp_y
			tar_left = tar_right = temp_x
			#copoints.erase(copoints.begin());
			copoints.pop(0)
			i=0
			while len(copoints)>i:
			#for i in range(len(copoints)):
				it=copoints[i]
				#print('current',it)
				distant = (temp_x - it[0])*(temp_x - it[0]) + (temp_y - it[1])*(temp_y - it[1])

				if distant <= r:#//根据点与类中心距离简单聚类

					temp_x = (temp_x*p_num + it[0]) / (p_num + 1)
					temp_y = (temp_y*p_num + it[1]) / (p_num + 1)

					tar_top =tar_top if tar_top < (it[1]) else it[1]
					tar_bottom = tar_bottom if tar_bottom > (it[1]) else it[1]
					tar_left =  tar_left if tar_left < (it[0]) else it[0]
					tar_right = tar_right if tar_right > (it[0]) else it[0]
					p_num+=1
					#it = copoints.erase(it)
					# print(i, len(copoints),copoints)
					# print('delete:',it)
					copoints.remove(it)
				else:
					i+=1
					continue

			if p_num >= 2:#//最起码要有两个角点才能被认为是目标
				target_pos.append([tar_left,tar_top,tar_right - tar_left,tar_bottom - tar_top])
				flag_target += 1
		return target_pos,flag_target
	else:
		flag_target = 0
		return target_pos,flag_target


def batch_horizon(dirname):
	import os
	saveDir = 'E:/HorizonLine'
	try:
		os.makedirs(saveDir)
	except:
		pass
	path = dirname#'E:/SeaShips_SMD/JPEGImages/'  # MVI_0790_VIS_OB_00291 006837 MVI_1627_VIS_00051.jpg
	paths = os.listdir(path)
	for p in tqdm(paths):
		basename = p
		img = cv2.imread(os.path.join(path, p))
		img = cv2.resize(img, None, None, fx=0.5, fy=0.5)
		cv2.imshow('src', img)
		cv2.waitKey(1)
		horizonLine = tar_detect(img, return_horizon=True)
		horizonLine = round(horizonLine)
		img = cv2.line(img, (0, horizonLine), (img.shape[1] - 1, horizonLine), (0, 0, 255), 2)
		#print(horizonLine)

		#海平线检测与焦点检测
		if False:
			tar_pos = tar_detect(img)
			for tar_p in tar_pos:
				tar_p = np.array(tar_p, dtype=np.int32)
				img = cv2.rectangle(img, (tar_p[0], tar_p[1]), (tar_p[0] + tar_p[2], tar_p[1] + tar_p[3]), (255, 0, 0))

		cv2.imshow('dst', img)
		cv2.imwrite(os.path.join(saveDir, basename), img)
		cv2.waitKey(1)

def one_horizon(path):

	# path = 'E:/SeaShips_SMD/JPEGImages/004555.jpg'# MVI_0790_VIS_OB_00291 006837 MVI_1627_VIS_00051.jpg
	#paths = os.listdir(path)
	#for p in paths:
	#basename = p
	img = cv2.imread(path)
	img = cv2.resize(img, None, None, fx=0.5, fy=0.5)
	cv2.imshow('src', img)
	cv2.waitKey(1)
	horizonLine = tar_detect(img, return_horizon=True)
	horizonLine = round(horizonLine)
	img = cv2.line(img, (0, horizonLine), (img.shape[1] - 1, horizonLine), (0, 0, 255), 2)
	print(horizonLine)
	tar_pos = tar_detect(img)
	for tar_p in tar_pos:
		tar_p = np.array(tar_p, dtype=np.int32)
		img = cv2.rectangle(img, (tar_p[0], tar_p[1]), (tar_p[0] + tar_p[2], tar_p[1] + tar_p[3]), (255, 0, 0))
	cv2.imshow('dst', img)
	#cv2.imwrite(os.path.join(saveDir, basename), img)
	cv2.waitKey(1)
if __name__=="__main__":
	path = 'E:/SeaShips_SMD/JPEGImages'#/004555.jpg
	# one_horizon(path)
	batch_horizon(path)