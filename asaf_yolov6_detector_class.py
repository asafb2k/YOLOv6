from tools.infer import Inferer
from pathlib import Path
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
import torch
import numpy as np
from yolov6.layers.common import DetectBackend
from yolov6.utils.events import LOGGER, load_yaml
import time
import os.path as osp
import os
import cv2
import math
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING']= '1'

class Yolov6Detector:
	def __init__(self, weights, device='0' ,img_size=640):
		self.__dict__.update(locals())
		self.btensorrt_inference = True if Path(weights).name.split('.')[1] == 'trt' or Path(weights).name.split('.')[1] == 'engine' else False
		self.weights_location = weights
		# Init model
		self.device = device
		self.img_size = img_size
		cuda = self.device != 'cpu' and torch.cuda.is_available()
		self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
		self.class_names = ['scoreboard']#load_yaml(yaml)['names']  # check image size
		if self.btensorrt_inference:
			self.batch_size = 1
			self.context, self.bindings, self.binding_addrs, trt_batch_size = self.init_engine(weights)
			assert trt_batch_size >= self.batch_size, f'The batch size you set is {self.batch_size}, it must <= tensorrt binding batch size {trt_batch_size}.'
			self.img_size = self.check_img_size(self.img_size)
		else:
			self.model = DetectBackend(weights, device=self.device)
			self.stride = self.model.stride
			self.img_size = self.check_img_size(self.img_size, s=self.stride)
			# Switch model to deploy status
			self.model_switch(self.model.model)
			# Half precision
			self.model.model.float()
			self.half = False
			if self.device.type != 'cpu':
				self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup
		#warmup:
		# for _ in range(10):
		# 	_,_ = self.detect(np.random.rand(500,500,3) * 255)

	def init_engine(self,engine):
		import tensorrt as trt
		from collections import namedtuple,OrderedDict
		Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
		logger = trt.Logger()
		trt.init_libnvinfer_plugins(logger, namespace="")
		with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
			self.model = runtime.deserialize_cuda_engine(f.read())
		bindings = OrderedDict()
		for index in range(self.model.num_bindings):
			name = self.model.get_binding_name(index)
			dtype = trt.nptype(self.model.get_binding_dtype(index))
			shape = tuple(self.model.get_binding_shape(index))
			data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
			bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
		binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
		context = self.model.create_execution_context()
		# warm up for 10 times
		tmp = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
		for _ in range(10):
			binding_addrs['images'] = int(tmp.data_ptr())
			context.execute_v2(list(binding_addrs.values()))
		return context, bindings, binding_addrs, self.model.get_binding_shape(0)[0]

	def make_divisible(self, x, divisor):
		# Upward revision the value x to make it evenly divisible by the divisor.
		return math.ceil(x / divisor) * divisor

	def check_img_size(self, img_size, s=32, floor=0):
		"""Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
		if isinstance(img_size, int):  # integer i.e. img_size=640
			new_size = max(self.make_divisible(img_size, int(s)), floor)
		elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
			new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
		else:
			raise Exception(f"Unsupported type of img_size: {type(img_size)}")

		if new_size != img_size:
			print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
		return new_size if isinstance(img_size,list) else [new_size]*2

	def model_switch(self, model):
		''' Model switch to deploy status '''
		from yolov6.layers.common import RepVGGBlock
		for layer in model.modules():
			if isinstance(layer, RepVGGBlock):
				layer.switch_to_deploy()

		LOGGER.info("Switch model to deploy modality.")

	def pre_process_image(self,img_src, img_size):
		'''Process image before image inference.'''
		image = letterbox(img_src, img_size)[0]
		# Convert
		image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
		image = torch.from_numpy(np.ascontiguousarray(image))
		image = image.to(self.device)
		image =  image.float()  # uint8 to fp16/32
		image /= 255  # 0 - 255 to 0.0 - 1.0
		image = image[None]
		return image, img_src


	def plot_box_and_label(self,image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
		# Add one xyxy box to image with label
		p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
		cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
		if label:
			tf = max(lw - 1, 1)  # font thickness
			w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
			outside = p1[1] - h - 3 >= 0  # label fits outside box
			p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
			cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
			cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,thickness=tf, lineType=cv2.LINE_AA)



	def rescale(self, ori_shape, boxes, target_shape):
			'''Rescale the output to the original image shape'''
			ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
			padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

			boxes[:, [0, 2]] -= padding[0]
			boxes[:, [1, 3]] -= padding[1]
			boxes[:, :4] /= ratio

			boxes[:, 0].clamp_(0, target_shape[1])  # x1
			boxes[:, 1].clamp_(0, target_shape[0])  # y1
			boxes[:, 2].clamp_(0, target_shape[1])  # x2
			boxes[:, 3].clamp_(0, target_shape[0])  # y2

			return boxes

	def box_convert(sef, x):
		# Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
		y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
		y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
		y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
		y[:, 2] = x[:, 2] - x[:, 0]  # width
		y[:, 3] = x[:, 3] - x[:, 1]  # height
		return y



	def detect(self,img_src,view_img = False):
		t1 = time.time()
		img, img_src = self.pre_process_image(img_src, self.img_size)
		t2 = time.time()
		if self.btensorrt_inference:
			t3 = time.time()
			self.binding_addrs['images'] = int(img.data_ptr())
			self.context.execute_v2(list(self.binding_addrs.values()))
			pred_results = self.bindings['outputs'][3]
			t4 = time.time()
		else:
			t3 = time.time()
			pred_results = self.model(img)
			t4 = time.time()
		t5 = time.time()
		det = non_max_suppression(pred_results, conf_thres=0.4, iou_thres=0.45, classes=None, max_det=1)[0]
		t6 = time.time()
		# check image and font
		img_ori = img_src
		assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
		det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
		if len(det) and view_img:
			for *xyxy, conf, cls in reversed(det):
				# if save_img:
				class_num = int(cls)  # integer class
				label =  f'{self.class_names[class_num]} {conf:.2f}'
				self.plot_box_and_label(img_ori, 1, xyxy, label, color=(255,0,0))

			img_src = np.asarray(img_ori)

			cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
			cv2.resizeWindow('test', img_src.shape[1], img_src.shape[0])
			cv2.imshow('test', img_src)
			cv2.waitKey(1)  # 1 millisecond
		
		if len(det) > 0:
			relative_detections = det[0][0:4] / self.img_size[0]
			return det[0][0:5],relative_detections
		else:
			return None, None

def run_on_directory_and_save_cropped(source_path: Path, destination_path: Path, detector:Yolov6Detector):
	for image_path in tqdm(list(source_path.rglob('*.png')), total=len(list(source_path.rglob('*.png')))):
		original_img = cv2.imread(str(image_path))
		img = original_img.copy()

		# detector returns bounding box as ratios
		bbox,_ = detector.detect(img, view_img=False)

		if bbox is not None:
			# convert bbox ratios to pixel values
			x1, y1, x2, y2 = [int(it) for it in bbox][:4]
			cropped_img = original_img[y1:y2, x1:x2]
			
			# find the relative path of the image with respect to the source root
			relative_image_path = image_path.relative_to(source_path)

			# create the output directory if it doesn't exist
			(destination_path / relative_image_path.parent).mkdir(parents=True, exist_ok=True)

			# save the cropped image to the destination path with the same name
			destination_img_path = destination_path / relative_image_path.parent / relative_image_path.name.replace('.png', f'_{img.shape[0]}x{img.shape[1]}.png')
			cv2.imwrite(str(destination_img_path), cropped_img)

if __name__ == "__main__":
	weights=Path(r"D:\best_stop_aug_ckpt.pt").as_posix()
	device = '0'
	img_size = 416

	# img = cv2.imread(Path(r"D:\cv-asaf-research\asaf scoreboard detector\darknet\ready_to_go_scoreboard_data_yolov4_TINY_07_08_2022_without_changing_the_cfg\val\6998831_10201_0.png").as_posix())

	detector = Yolov6Detector(weights, device, img_size)
	# a, b = detector.detect(img, view_img=False)
	run_on_directory_and_save_cropped(Path(r"D:\cv-asaf-research\unsupervised_clustering\input\tennis july 6\tennis_july_6_for_asaf\tennis_july_6_for_asaf\Tennis"), Path(r"D:\Unsupervised-Classification\input\Cropped"), detector)