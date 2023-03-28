

def get_gpu_mem_info(gpu_id=0):
  import pynvml
  pynvml.nvmlInit()
  if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
    print('gpu_id {} not found!'.format(gpu_id))
    return 0, 0, 0

  handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
  meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
  total = round(meminfo.total / 1024 / 1024, 2)
  used = round(meminfo.used / 1024 / 1024, 2)
  free = round(meminfo.free / 1024 / 1024, 2)

  print('gpu consume：total {} MB， used {} MB， free {} MB'.format(total, used, free))
  # return total, used, free


class Dict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

def dict_to_object(dictObj):
  if not isinstance(dictObj, dict):
    return dictObj
  inst=Dict()
  for k,v in dictObj.items():
    inst[k] = dict_to_object(v)
  return inst