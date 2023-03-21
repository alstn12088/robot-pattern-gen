import torch
import numpy as np
from torchvision.utils import save_image
image = np.load('data/version1/input_1.npy')
score = np.load('data/version1/output_1.npy')

s = np.arange(image.shape[0])
np.random.shuffle(s)

image = image[s]
score = score[s]


train_image = image[:int(image.shape[0]*0.9)]
test_image = image[int(image.shape[0]*0.9):]

train_score = score[:int(image.shape[0]*0.9)]
test_score = score[int(image.shape[0]*0.9):]


save_image(torch.from_numpy(train_image).view(-1, 1, 30, 30), './samples/train_' + '.png')




# np.save('data/train/train_image.npy',train_image)
# np.save('data/train/train_score.npy',train_score)

# np.save('data/test/test_image.npy',test_image)
# np.save('data/test/test_score.npy',test_score)

# np.save('data/all/image.npy',image)
# np.save('data/all/score.npy',score)
