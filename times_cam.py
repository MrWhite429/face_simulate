import cv2
import numpy as np
import skimage.transform as tf
from seetaface.api import *

check_face_img = {
	"asserts/tomori_face.jpg": "Tomori",
	"asserts/chan.jpg": "Chan",
	"asserts/tao.jpg": "Tao",
}
faces = {}

init_mask = FACE_TRACK | FACE_DETECT | LANDMARKER5
seetaFace = SeetaFace(init_mask)

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp
import torch
from newdemo import make_animation, load_checkpoints
from copy import deepcopy

cap = cv2.VideoCapture(0)


def get_mark(frame):
	detect_result = seetaFace.Track(frame)
	face = detect_result.data[0].pos
	if detect_result.size < 1:
		# print('no face in image')
		return None
	points_5 = seetaFace.mark5(frame, face)
	return deepcopy(face), points_5


def show_mark(frame, src_face, points_5, wt, ht,col=(0,0,255)):
	cv2.rectangle(frame, (int(src_face.x * wt), int(src_face.y * ht)),
	              (int((src_face.x + src_face.width) * wt), int((src_face.y + src_face.height) * ht)),
	              col, 2)
	for i in range(5):
		cv2.circle(frame, (int(points_5[i].x * wt), int(points_5[i].y * ht)), 5, col, -1)


def dis(pa,pb):
	return (pa[0]-pb[0])**2+(pa[1]-pb[1])**2

def get_points(face,shape):
	x1,y1 = face.x,face.y
	x2,y2 = x1+face.width,y1+face.height
	return (x1/shape[0],y1/shape[1]),(x2/shape[0],y2/shape[1])

def get_similar(face_a,shape_a,face_b,shape_b):
	pa1,pa2 = get_points(face_a,shape_a)
	pb1,pb2 = get_points(face_b,shape_b)
	return dis(pa1,pb1)+dis(pa2,pb2)


if __name__ == '__main__':
	config = "./config/vox-adv-256.yaml"
	checkpoint = "./checkpoint/vox-adv-cpk.pth.tar"
	# imgp = './data/risol.png'
	imgp = './data/polpot.jpg'

	source_image = cv2.imread(imgp)
	src_shape = source_image.shape
	wt, ht = 256 / src_shape[0], 256 / src_shape[1]
	print('init model')
	generator, kp_detector = load_checkpoints(config_path=config, checkpoint_path=checkpoint, cpu=False)
	print('generator')
	src_face, points_5 = get_mark(source_image)
	source_image = tf.resize(source_image, (256, 256))
	source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
	source = source.cuda()
	kp_source = kp_detector(source)
	kp_driving_initial = None
	in_match = False

	while True:
		ret, frame = cap.read()
		frame = frame[:, 90:-90]
		frame = cv2.flip(frame, 1)
		now_shape = frame.shape
		if not in_match:
			now_res = get_mark(frame)
			if now_res is not None:
				now_face, nowpt5 = now_res
				sim_res = get_similar(now_face,now_shape,src_face,src_shape)
				print(sim_res)
				if sim_res<0.003:
					in_match = True
			frame = tf.resize(frame, (256, 256))
			show_mark(frame, src_face, points_5, wt, ht,(255,0,0))
			if now_res is not None:
				show_mark(frame, now_face, nowpt5, 256 / now_shape[0], 256 / now_shape[1],(0,0,255))
			frameres = source_image
		else:
			frame = tf.resize(frame, (256, 256))
			driving_frame = torch.tensor(frame[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
			driving_frame = driving_frame.cuda()
			kp_driving = kp_detector(driving_frame)
			if kp_driving_initial == None:
				kp_driving_initial = kp_detector(driving_frame)
			kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
			                       kp_driving_initial=kp_driving_initial, use_relative_movement=True,
			                       use_relative_jacobian=True, adapt_movement_scale=True)
			out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
			frameres = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
		frame = np.hstack([frame, source_image, frameres])
		cv2.imshow('Local Camera', frame)

		if cv2.waitKey(25) & 0xFF == ord('1'):
			if not in_match:
				in_match = True
			else:
				kp_driving_initial = kp_detector(driving_frame)
		if cv2.waitKey(1) & 0xFF == ord('2'):
			break

	cap.release()
	cv2.destroyAllWindows()
