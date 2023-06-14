import torch
import numpy as np
from torchvision.utils import save_image
image = np.load('data/version2/Input_data_manual_1_100_50.npy')
score = np.load('data/version2/Output_data_manual_1_2.npy')

s = np.arange(image.shape[0])
np.random.shuffle(s)

image = image[s]
score = score[s]


print(score)

# train_image = image[:int(image.shape[0]*0.9)]
# test_image = image[int(image.shape[0]*0.9):]

# train_score = score[:int(image.shape[0]*0.9)]
# train_score = 0.5*train_score[:,0] + 0.5*train_score[:,1]
# train_score = train_score.reshape(-1,1)


# test_score = score[int(image.shape[0]*0.9):]
# test_score = 0.5*test_score[:,0] + 0.5*test_score[:,1]
# test_score = test_score.reshape(-1,1)


#save_image(torch.from_numpy(train_image).view(-1, 1, 30, 30), './samples/train_' + '.png')




# np.save('data/version2/train/train_image.npy',train_image)
# np.save('data/version2/train/train_score.npy',train_score)

# np.save('data/version2/test/test_image.npy',test_image)
# np.save('data/version2/test/test_score.npy',test_score)

# np.save('data/all/image.npy',image)
# np.save('data/all/score.npy',score)
