# Author: Ziga Trojer
import numpy as np
import math
import os.path

files_list=[
	'Beaker_1','Beaker_2','Beaker_3','Beaker_4','Beaker_5','Beaker_6','Beaker_7','Beaker_8','Beaker_9','Beaker_10','Beaker_11','Beaker_12','Beaker_13','Beaker_14','Beaker_15',
	'BubbleBalloon_1','BubbleBalloon_2','BubbleBalloon_3','BubbleBalloon_4','BubbleBalloon_5','BubbleBalloon_6','BubbleBalloon_7','BubbleBalloon_8','BubbleBalloon_9',
	'BubbleBalloon_10','BubbleBalloon_11','BubbleBalloon_12','BubbleBalloon_13','BubbleBalloon_14','BubbleBalloon_15','Bulb_1','Bulb_2','Bulb_3','Bulb_4','Bulb_5',
	'Bulb_6','Bulb_7','Bulb_8','Bulb_9','Bulb_10','Bulb_11','Bulb_12','Bulb_13','Bulb_14','Bulb_15','Flask_1','Flask_2','Flask_3','Flask_4','Flask_5','Flask_6','Flask_7',
	'Flask_8','Flask_9','Flask_10','Flask_11','Flask_12','Flask_13','Flask_14','Flask_15','GlassBall_1','GlassBall_2','GlassBall_3','GlassBall_4','GlassBall_5','GlassBall_6',
	'GlassBall_7','GlassBall_8','GlassBall_9','GlassBall_10','GlassBall_11','GlassBall_12','GlassBall_13','GlassBall_14','GlassBall_15','GlassBottle_1','GlassBottle_2',
	'GlassBottle_3','GlassBottle_4','GlassBottle_5','GlassBottle_6','GlassBottle_7','GlassBottle_8','GlassBottle_9','GlassBottle_10','GlassBottle_11','GlassBottle_12',
	'GlassBottle_13','GlassBottle_14','GlassBottle_15','GlassCup_1','GlassCup_2','GlassCup_3','GlassCup_4','GlassCup_5','GlassCup_6','GlassCup_7','GlassCup_8','GlassCup_9',
	'GlassCup_10','GlassCup_11','GlassCup_12','GlassCup_13','GlassCup_14','GlassCup_15','GlassJar_1','GlassJar_2','GlassJar_3','GlassJar_4','GlassJar_5','GlassJar_6',
	'GlassJar_7','GlassJar_8','GlassJar_9','GlassJar_10','GlassJar_11','GlassJar_12','GlassJar_13','GlassJar_14','GlassJar_15','GlassSlab_1','GlassSlab_2','GlassSlab_3',
	'GlassSlab_4','GlassSlab_5','GlassSlab_6','GlassSlab_7','GlassSlab_8','GlassSlab_9','GlassSlab_10','GlassSlab_11','GlassSlab_12','GlassSlab_13','GlassSlab_14',
	'GlassSlab_15','JuggleBubble_1','JuggleBubble_2','JuggleBubble_3','JuggleBubble_4','JuggleBubble_5','JuggleBubble_6','JuggleBubble_7','JuggleBubble_8','JuggleBubble_9',
	'JuggleBubble_10','JuggleBubble_11','JuggleBubble_12','JuggleBubble_13','JuggleBubble_14','JuggleBubble_15','MagnifyingGlass_1','MagnifyingGlass_2','MagnifyingGlass_3',
	'MagnifyingGlass_4','MagnifyingGlass_5','MagnifyingGlass_6','MagnifyingGlass_7','MagnifyingGlass_8','MagnifyingGlass_9','MagnifyingGlass_10','MagnifyingGlass_11',
	'MagnifyingGlass_12','MagnifyingGlass_13','MagnifyingGlass_14','MagnifyingGlass_15','ShotGlass_1','ShotGlass_2','ShotGlass_3','ShotGlass_4','ShotGlass_5','ShotGlass_6',
	'ShotGlass_7','ShotGlass_8','ShotGlass_9','ShotGlass_10','ShotGlass_11','ShotGlass_12','ShotGlass_13','ShotGlass_14','ShotGlass_15','TransparentAnimal_1','TransparentAnimal_2',
	'TransparentAnimal_3','TransparentAnimal_4','TransparentAnimal_5','TransparentAnimal_6','TransparentAnimal_7','TransparentAnimal_8','TransparentAnimal_9','TransparentAnimal_10',
	'TransparentAnimal_11','TransparentAnimal_12','TransparentAnimal_13','TransparentAnimal_14','TransparentAnimal_15','WineGlass_1','WineGlass_2','WineGlass_3','WineGlass_4',
	'WineGlass_5','WineGlass_6','WineGlass_7','WineGlass_8','WineGlass_9','WineGlass_10','WineGlass_11','WineGlass_12','WineGlass_13','WineGlass_14','WineGlass_15','WubbleBubble_1',
	'WubbleBubble_2','WubbleBubble_3','WubbleBubble_4','WubbleBubble_5','WubbleBubble_6','WubbleBubble_7','WubbleBubble_8','WubbleBubble_9','WubbleBubble_10','WubbleBubble_11',
	'WubbleBubble_12','WubbleBubble_13','WubbleBubble_14','WubbleBubble_15'
	]

def read_meta_info(subdirectory):
	meta_data_file = os.path.join(subdirectory + 'meta_info.txt')
	with open(meta_data_file) as f:
		lines = f.readlines()
	for line in lines:
		if 'Length' in line:
			seq_len = [int(s) for s in line.split() if s.isdigit()][0]
	return seq_len

class Anchor:
	def __init__(self, seq_len, subdirectory):
		self.stride = 50
		self.seq_len = seq_len
		self.subdirectory = subdirectory

	def generate_anchor(self):
		anchors = np.zeros((self.seq_len)) * 1.0
		half = math.floor(len(anchors) / 2)
		anchors[0] = 1
		anchors[-1] = -1
		anchors[::self.stride] = 1
		#anchors[:half:self.stride] = 1
		#anchors[half + (half % self.stride)-1::self.stride] = -1
		anchors[half:] = anchors[half:] * -1
		with open(os.path.join(subdirectory + 'anchor.value'), 'w') as the_file:
			for anchor_val in anchors:
				if anchor_val == 0:
					the_file.write(str(abs(anchor_val)) + '\n')
				else:
					the_file.write(str(anchor_val) + '\n')

if __name__ == '__main__':
	for file in files_list:
		subdirectory = file + '/'
		seq_length = read_meta_info(subdirectory)
		anchors = Anchor(seq_length, subdirectory)
		anchors.generate_anchor()
		print(f'Created anchor file for {file}, sequence length: {seq_length}')
