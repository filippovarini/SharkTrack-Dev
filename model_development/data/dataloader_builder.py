from torch.utils.data import DataLoader, random_split
from data.dataset import Dataset

TRAIN_RATIO = 1
VAL_RATIO = 0.0  # validation from out-of-sample data
TEST_RATIO = 0.0 # test with out-of-sample data

class DataLoaderBuilder():
  def __init__(self, dataset: Dataset, batch_size):
    self.dataset = dataset
    self.train_size = int(TRAIN_RATIO * len(dataset))
    self.val_size = int(VAL_RATIO * len(dataset))
    self.test_size = len(dataset) - self.train_size - self.val_size
    self.batch_size = batch_size

  @staticmethod
  def get_split_ratios():
    return TRAIN_RATIO, VAL_RATIO, TEST_RATIO
  
  def build(self):
    train_dataset, val_dataset, test_dataset = random_split(self.dataset, [self.train_size, self.val_size, self.test_size])
    train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
    return train_loader, val_loader, test_loader