import cv2
import numpy as np
import tensorflow as tf

import argparse
import os
import random

__version__ = "1.0.0"
__image_classes__ = ["0", "1"]
__image_types__ = [".png", ".bmp", ".jpg"]
__data_shape__ = (100, 100, 1)
__batch_size__ = 32
__train_steps__ = 100
__test_steps__ = 10

class ImageReader:
	def __init__(self, path, **kwargs):
		self.path = path
		self.classes = kwargs.get("classes", __image_classes__)
		self.shape = kwargs.get("shape", __data_shape__)
		self.types = kwargs.get("types", __image_types__)
	
	def _find_images(self):
		images = []
		for i, cls in enumerate(self.classes):
			path = os.path.join(self.path, cls)
			for obj in os.listdir(path):
				fn = os.path.join(path, obj)
				if os.path.isfile(fn) and os.path.splitext(obj)[1] in self.types:
					images.append((fn, i))
		return images
	
	def _convert(self, im):
		w, h = im.shape[1], im.shape[0]
		nw, nh = self.shape[1], self.shape[0]
		
		# Resize
		ratio = h / w
		tmpw = max(nw, int(round(nw/ratio)))
		tmph = max(nh, int(round(nh*ratio)))
		im_resize = cv2.resize(im, (tmpw, tmph), interpolation=cv2.INTER_LANCZOS4)
		
		# Crop
		x = tmpw//2 - nw//2
		y = tmph//2 - nh//2
		im_crop = im_resize[y:y+nh, x:x+nw]
		
		# Reshape and normalize
		data = im_crop.reshape(self.shape).astype(np.float32)
		return data / 255.0
	
	def _generator(self):
		images = self._find_images()
		random.shuffle(images)
		for file, cls in images:
			im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
			data = self._convert(im)
			yield data, np.array([cls], dtype=np.int32)
	
	def dataset(self):
		return tf.data.Dataset.from_generator(
			self._generator,
			(tf.float32, tf.int32),
			output_shapes=(self.shape, (1))
		)

def icemet_densenet201():
	densenet = tf.keras.applications.densenet.DenseNet201(
		include_top=False,
		weights=None,
		input_tensor=None,
		input_shape=__data_shape__,
		pooling="avg",
		classes=None
	)
	inputs = densenet.layers[0].input
	outputs = tf.keras.layers.Dense(1, activation="sigmoid")(densenet.layers[-1].output)
	return tf.keras.models.Model(inputs=inputs, outputs=outputs)

def train_main():
	parser = argparse.ArgumentParser("ICEMET neural network trainer")
	parser.add_argument("-id", "--image_dir", type=str, default="images", help="image directory", metavar="str")
	parser.add_argument("-gd", "--graph_dir", type=str, default="graph", help="graph directory", metavar="str")
	parser.add_argument("-md", "--model_dir", type=str, default="model", help="model directory", metavar="str")
	parser.add_argument("-e", "--epochs", type=int, default=10000, help="number of epochs", metavar="int")
	parser.add_argument("-ci", "--checkpoint_interval", type=int, default=-1, help="Checkpoint interval", metavar="int")
	parser.add_argument("-V", "--version", action="store_true", help="print version information")
	args = parser.parse_args()
	
	reader = ImageReader(args.image_dir)
	dataset = reader.dataset().batch(__batch_size__).repeat()
	
	model = icemet_densenet201()
	model.compile(
		optimizer=tf.keras.optimizers.Adam(lr=0.001),
		loss="binary_crossentropy",
		metrics=["accuracy"]
	)
	
	if not os.path.exists(args.model_dir):
		os.makedirs(args.model_dir)
	with open(os.path.join(args.model_dir, "icemet-nn.json"), "w") as fp:
		fp.write(model.to_json())
	model_file = os.path.join(args.model_dir, "icemet-nn_{epoch}.h5")
	
	callbacks = [
		tf.keras.callbacks.TensorBoard(
			log_dir=args.graph_dir,
			write_graph=True,
			write_images=True
		)
	]
	if args.checkpoint_interval > 0:
		callbacks.append(tf.keras.callbacks.ModelCheckpoint(
			model_file,
			period=args.checkpoint_interval,
			save_weights_only=False,
			verbose=1
		))
	
	model.fit(dataset, epochs=args.epochs, steps_per_epoch=__train_steps__, callbacks=callbacks)
	model.save(model_file.format(epoch=args.epochs))

def test_main():
	parser = argparse.ArgumentParser("ICEMET neural network testing")
	parser.add_argument("-m", "--model", type=str, default="icemet_nn.h5", help="model file", metavar="str")
	parser.add_argument("-id", "--image_dir", type=str, default="images", help="image directory", metavar="str")
	args = parser.parse_args()
	
	reader = ImageReader(args.image_dir)
	dataset = reader.dataset().batch(__batch_size__)
	
	model = tf.keras.models.load_model(args.model)
	model.compile(
		optimizer=tf.keras.optimizers.Adam(lr=0.001),
		loss="binary_crossentropy",
		metrics=["accuracy"]
	)
	model.evaluate(dataset, steps=__test_steps__)
