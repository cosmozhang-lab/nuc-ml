import utils
import numpy as np
import random

image_height = 40
image_width = 2000
image_channels = 2

class NucBatch:
  def __init__(self, data=None, labels=None):
    self.data = data
    self.labels = labels

class NucDataset:
  def __init__(self, filenameOrInstance, matter=None, one_hot=True, reshape=False):
    if isinstance(filenameOrInstance,NucDataset):
      inst = filenameOrInstance
      self.matter = inst.matter
      self.data = inst.data
      self.labels = inst.labels
      self.one_hot = inst.one_hot
      self.flatten = inst.flatten
    else:
      filename = filenameOrInstance
      self.matter = matter
      self.data = None
      self.labels = None
      self.one_hot = one_hot
      self.flatten = reshape
      if not filename is None:
        dict_obj = utils.loadDataset(filename)
        data = dict_obj['data']
        labels = dict_obj['labels']
        sample_number = len(data)
        # here load the transposed images (height as width, width as height)
        data_array = np.zeros([sample_number, image_height*image_width*image_channels]) if self.flatten else np.zeros([sample_number, image_height, image_width, image_channels])
        labels_array = np.zeros([sample_number, 1]) if self.one_hot else np.zeros([sample_number])
        for i in range(sample_number):
          if self.one_hot:
            labels_array[i,0] = float(labels[i])
          else:
            labels_array[i] = float(labels[i])
          dataitem = np.zeros([image_height,image_width,image_channels])
          for ch in xrange(image_channels): dataitem[:,:,ch] = np.transpose(data[i][:,:,ch])
          if self.flatten:
            data_array[i,:] = np.reshape(dataitem,[image_height*image_width*image_channels])
          else:
            data_array[i,:,:,:] = dataitem
        self.data = data_array.astype(np.float32)
        self.labels = labels_array.astype(np.float32)
      self._renew()

  def extend(self, dataset):
    if type(dataset) == str:
      self.extend(NucDataset(filename=dataset, one_hot=self.one_hot, reshape=self.flatten))
    elif isinstance(dataset, NucDataset) and (dataset.sample_number > 0):
      new_data = dataset.data
      new_labels = dataset.labels
      if self.one_hot and (not dataset.one_hot):
        new_new_labels = np.zeros([dataset.sample_number,1])
        new_new_labels[:,0] = new_labels[i]
        new_labels = new_new_labels
      elif (not self.one_hot) and dataset.one_hot:
        new_new_labels = np.zeros([dataset.sample_number])
        new_new_labels[:] = new_labels[:,0]
        new_labels = new_new_labels
      if self.flatten and (not dataset.flatten):
        new_data = np.reshape(new_data, [dataset.sample_number, image_height*image_width*image_channels])
      elif (not self.flatten) and dataset.flatten:
        new_data = np.reshape(new_data, [dataset.sample_number, image_height, image_width, image_channels])
      self.data = new_data if (self.data is None) else np.append(self.data, new_data, axis=0)
      self.labels = new_labels if (self.labels is None) else np.append(self.labels, new_labels, axis=0)
      self._renew()

  def _renew(self):
    self._data = None
    self._labels = None
    self.sample_number = 0 if (self.data is None) else len(self.data)
    # to force refresh next time
    self._epoches_completed = -1
    self._index_in_epoch = self.sample_number + 1

  def next_batch(self, batch_size):
    if self._index_in_epoch + batch_size > self.sample_number:
      lst = range(self.sample_number)
      random.shuffle(lst)
      self._data = self.data[lst,:] if self.flatten else self.data[lst,:,:,:]
      self._labels = self.labels[lst,:] if self.one_hot else self.labels[lst]
      self._epoches_completed += 1
      self._index_in_epoch = 0
    start = self._index_in_epoch
    end = self._index_in_epoch + batch_size
    self._index_in_epoch = end
    batch_data = self._data[start:end,:] if self.flatten else self._data[start:end,:,:,:]
    batch_labels = self._labels[start:end,:] if self.one_hot else self._labels[start:end]
    return NucBatch(data=batch_data, labels=batch_labels)

  def shuffle(self):
    if self.sample_number == 0:
      return
    seq = range(self.sample_number)
    random.shuffle(seq)
    self.select(seq)

  def select(self,idx):
    if self.flatten:
      self.data = self.data[idx,:]
    else:
      self.data = self.data[idx,:,:,:]
    if self.one_hot:
      self.labels = self.labels[idx,:]
    else:
      self.labels = self.labels[idx]
    self._renew()

class Nuc:
  def __init__(self, filename):
    testr = 0.2
    whole = NucDataset(filename)
    whole.shuffle()
    testn = int(whole.sample_number * testr)
    testset = NucDataset(whole)
    testset.select(range(0,testn))
    trainset = NucDataset(whole)
    trainset.select(range(testn,whole.sample_number))
    self.test = testset
    self.train = trainset
