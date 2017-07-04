# -*- coding: utf-8

import scipy as sp
import numpy as np
import pickle as pk
import sys,os
import re
import progressbar
import struct

DMT = DM_TIME = 2000
DMR = DM_REPEAT = 40

def read_csv(filepath, delimiter=",", fmt=None):
  f = open(filepath, 'r')
  content = []
  while True:
    l = f.readline()
    if len(l) == 0:
      break
    l = l.rstrip()
    l = l.replace("\xef\xbb\xbf","") # remove the BOM header of utf-8
    if len(l) == 0:
      continue
    cs = l.split(delimiter)
    if fmt:
      cs = [fmt(c) for c in cs]
    content.append(cs)
  f.close()
  return content

class RecordItem:
  def __init__(self, filename=None, name=None, matter=None, isPositive=False, isTrue=False):
    if filename:
      self.pos = np.array(read_csv(filename + "_Pos.csv", fmt=int))[0:DMT,0:DMR]
      self.neg = np.array(read_csv(filename + "_Neg.csv", fmt=int))[0:DMT,0:DMR]
      if self.pos.shape[0] != DMT or self.pos.shape[1] != DMR:
        raise Exception("Unexpected positive curve matrix size, need <%dx%d>, but <%dx%d> is given." % (DMT,DMR,self.pos.shape[0],self.pos.shape[1]))
      if self.neg.shape[0] != DMT or self.neg.shape[1] != DMR:
        raise Exception("Unexpected negative curve matrix size, need <%dx%d>, but <%dx%d> is given." % (DMT,DMR,self.neg.shape[0],self.neg.shape[1]))
    else:
      self.pos = self.neg = None
    self.name = name
    self.matter = matter
    self.isPositive = isPositive
    self.isTrue = isTrue

  def __getattr__(self, name):
    if name == "positive":
      return (self.isPositive and self.isTrue) or ((not self.isPositive) and (not self.isTrue))
    else:
      return None

def readDatas(withLog=False, withProgress=True, taskFilter=None):
  tasks = []
  tasks.append( ("../../HMTD正常报警.csv", "HMTD", True, True) )
  tasks.append( ("../../HMTD该报未报.csv", "HMTD", False, False) )
  tasks.append( ("../../HMTD误报.csv", "HMTD", True, False) )
  tasks.append( ("../../TNT正确报警.csv", "TNT", True, True) )
  tasks.append( ("../../TNT误报.csv", "TNT", True, False) )

  if not taskFilter is None:
    filteredTasks = []
    for task in tasks:
      if taskFilter(matter=task[1], isPositive=task[2], isTrue=task[3]):
        filteredTasks.append(task)
    tasks = filteredTasks

  matterNames = []
  for task in tasks:
    if not task[1] in matterNames:
      matterNames.append(task[1])

  datas = {}
  for mn in matterNames:
    datas[mn] = []

  namereg = re.compile(r"([\w\W]+)_(Pos|Neg)\.csv")

  total = 0
  for task in tasks:
    dirname = os.path.join( os.path.dirname(__file__), task[0] )
    filenames = os.listdir(dirname)
    total += len(filenames)

  if withProgress:
    prompt = withProgress if isinstance(withProgress,str) else "Reading"
    widgets = ["%s: "%prompt, progressbar.Percentage(), ' ', progressbar.Bar()]
    pbar = progressbar.ProgressBar(widgets=widgets, maxval=total).start()

  cnt = 0
  for task in tasks:
    dirname = os.path.join( os.path.dirname(__file__), task[0] )
    matter = task[1]
    isPositive=task[2]
    isTrue=task[3]
    filenames = os.listdir(dirname)
    recordnames = []
    for fn in filenames:
      m = namereg.match(fn)
      if m is None:
        continue
      name = m.group(1)
      if name in recordnames:
        continue
      recordnames.append(name)
    for name in recordnames:
      fn = os.path.join(dirname,name)
      record = RecordItem(filename=fn, name=name, matter=matter, isPositive=isPositive, isTrue=isTrue)
      datas[matter].append(record)
      # progress printing
      cnt += 2
      pg = int(float(cnt)*100/float(total))
      if withProgress:
        pbar.update(cnt)
      elif withLog:
        print ">> Read: %s (%d%%)" % fn

  # progress printing
  if withProgress:
    pbar.finish()

  return datas

def singleMatterFilter(matterName):
  def theFilter(matter,isPositive,isTrue):
    return matter == matterName
  return theFilter



def genDataset(filename, matterName):
  data = readDatas(taskFilter=singleMatterFilter(matterName))[matterName]
  total = len(data)
  f = open(filename, "wb")
  f.write(struct.pack("i",total))
  f.write(struct.pack("i",DMT))
  f.write(struct.pack("i",DMR))

  pbar = progressbar.ProgressBar(widgets=["Saving labels: ", progressbar.Counter(),"/%d"%total, ' ', progressbar.Bar()], maxval=total).start()
  cnt = 0
  for i in range(total):
    f.write(struct.pack("i",int(data[i].positive)))
    cnt += 1
    pbar.update(cnt)
  pbar.finish()

  pbar = progressbar.ProgressBar(widgets=["Saving datas: ", progressbar.Counter(),"/%d"%total, ' ', progressbar.Bar()], maxval=total).start()
  cnt = 0
  for i in range(total):
    arr = data[i].pos.reshape([DMT*DMR])
    for j in range(DMT*DMR):
      f.write(struct.pack("i",int(arr[j])))
    arr = data[i].neg.reshape([DMT*DMR])
    for j in range(DMT*DMR):
      f.write(struct.pack("i",int(arr[j])))
    cnt += 1
    pbar.update(cnt)
  pbar.finish()

  f.close()

def loadDataset(filename):
  f = open(filename, "rb")
  h = struct.unpack("3i", f.read(3*4))
  total = h[0]

  print "Loading labels..."
  labels = struct.unpack("%di"%total, f.read(total*4))

  pbar = progressbar.ProgressBar(widgets=["Loading datas: ", progressbar.Counter(),"/%d"%total, ' ', progressbar.Bar()], maxval=total).start()
  cnt = 0
  datas = [None for i in range(total)]
  for i in range(total):
    arr = np.zeros([DMT,DMR,2],int)
    arr[:,:,0] = np.array(struct.unpack("%di"%(DMT*DMR), f.read(DMT*DMR*4))).reshape([DMT,DMR])
    arr[:,:,1] = np.array(struct.unpack("%di"%(DMT*DMR), f.read(DMT*DMR*4))).reshape([DMT,DMR])
    datas[i] = arr
    cnt += 1
    pbar.update(cnt)
  pbar.finish()

  f.close()

  return { "labels": labels, "data": datas }

# save dataset to mat file
def saveDataset(filename_bin, filename_mat):
  from nucdata import NucDataset
  from scipy.io import savemat
  dataset = NucDataset(filename_bin, one_hot=False, reshape=False)
  print "Saving dataset ..."
  savemat(filename_mat, { "labels": dataset.labels, "data": dataset.data })
  print "Save dataset OK"



if __name__ == "__main__":
  import time
  t1 = time.time()
  if len(sys.argv) > 1:
    if sys.argv[1] == "readcsvonly":
      readDatas(taskFilter=singleMatterFilter("TNT"))
    if sys.argv[1] == "gendataset":
      genDataset("../../datasets/TNT.bin", "TNT")
      genDataset("../../datasets/HMTD.bin", "HMTD")
    if sys.argv[1] == "loaddataset":
      loadDataset("../../datasets/TNT.bin")
      loadDataset("../../datasets/HMTD.bin")
    if sys.argv[1] == "savedataset":
      saveDataset("../../datasets/TNT.bin", "../../datasets/TNT.mat")
      saveDataset("../../datasets/HMTD.bin", "../../datasets/HMTD.mat")
  t2 = time.time()
  print "Elapsed: %d s" % int(t2-t1)
