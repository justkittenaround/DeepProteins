##looks through antibody database, remove if string length is bigger than.

import os
from PIL import Image
import numpy as np
os.chdir('/home/whale/Desktop/Rachel/DeepProteins/AutoAntibodies/cyclegan/data/')
counts = {}
def search(folder):
	files = os.listdir(folder)
	for filename in files:
		filepath = os.path.join(folder, filename)
		im = np.asarray(Image.open(filepath)) # read in the image
		im = np.sum(im, 2)
		im_valid = np.argmax(im, 0)  # find out which letter is at that position
		string = im[:, (im_valid != 23.)]  # take the part of the image where the number is not 23
		stringlen = string.shape[1]  # get the length of the valid string
		print(filepath)
		if stringlen > 256:
			os.remove(filepath)
			print(filename)
		print('meowzers')
search('testA')
search('testB')
search('trainA')
search('trainB')
