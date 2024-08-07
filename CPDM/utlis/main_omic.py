import torch, os, sys
from utlis.multiOmic import train_pdm, Unet, get_dataloader, prepareDataset
from utlis.crossVal import generateSubsets
# try inner product?

rootPath = '/data/lab_ph/kyle/projects/BreastCancer'
# rootPath = '/content/drive/MyDrive/BreastCancer'

def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

modelName = 'omic_bs10_1_1.pt'
checkpointPath = os.path.abspath('../BreastCancer/checkpoints')
fullCheckpointPath = checkpointPath + '/' + modelName

config = {
    'use_checkpoint' : False,
    'checkpoint_path': fullCheckpointPath,
    'epochs' : 1100,
    'loss_type' : 'l2',
    'timesteps' : 1500,
    'image_size' : 128,
    'channels' : 1,
    'batch_size' : 3,
    'lr' : 1e-4,
    'SideImgPath' : os.path.abspath('../BreastCancer/data/SidePNG128'),
    'SideImgIDpath' : find_file('SideViewBTF.csv', rootPath),
    'multiOmicPath' : find_file('BTF_features.csv', rootPath),
    'test' : [1,16,24,33]
}

if __name__ == "__main__":
  subsets = generateSubsets()
  print('Start multi-omic ....')
  for i, each in enumerate(subsets):
    config['test'] = each
    config['checkpoint_path'] = checkpointPath + '/' + f'omic_bs3_Folder {i}.pt'
    train_pdm(config)
    print(f'Finished Folder {i}')

  

