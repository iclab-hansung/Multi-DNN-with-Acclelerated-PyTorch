import torch
import time
import torchvision.models as models
from threading import Thread

GPU_NUM = 0 
device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

dense_num = 0 
res_num = 0
alex_num = 0
vgg_num = 0
wide_num = 0
squeeze_num = 0
mobile_num = 0
mnas_num = 0
inception_num = 0
shuffle_num = 16 
resx_num = 0

def predict(model, inputs):
    #start = time.time()
    out = model(inputs)
    #end = time.time()
    print(out[0,0])
    print(model.__class__.__name__)
    return (model.__class__.__name__)

inputs = torch.ones(1,3,224,224).cuda()
inputs2 = torch.ones(1,3, 299, 299).cuda()

# model load
inception = models.inception_v3(pretrained=True).cuda().eval()
squeeze = models.squeezenet1_0(pretrained=True).cuda().eval()
shuffle = models.shufflenet_v2_x1_0(pretrained=True).cuda().eval()
alex = models.alexnet(pretrained=True).cuda().eval()
vgg = models.vgg16(pretrained=True).cuda().eval()
dense = models.densenet201(pretrained=True).cuda().eval()
res = models.resnet152(pretrained=True).cuda().eval()
mnas = models.mnasnet1_0(pretrained=True).cuda().eval()
mobile = models.mobilenet_v2(pretrained=True).cuda().eval()
resx = models.resnext101_32x8d(pretrained=True).cuda().eval()
wide = models.wide_resnet50_2(pretrained=True).cuda().eval()

# WARM UP

for i in range(4):
    dense(inputs)

for i in range(4):
    res(inputs)

for i in range(4):
    alex(inputs)

for i in range(4):
    vgg(inputs)

for i in range(4):
    wide(inputs)

for i in range(4):
    squeeze(inputs)

for i in range(4):
    mobile(inputs)

for i in range(4):
    mnas(inputs)

for i in range(4):
    inception(inputs2)

for i in range(4):
    shuffle(inputs)

for i in range(4):
    resx(inputs)
torch.cuda.synchronize()
print("warm up end")


model_list=[] 
inception_list=[]

for i in range(dense_num):
    model_list.append(dense)

for i in range(res_num):
    model_list.append(res)

for i in range(alex_num):
    model_list.append(alex)

for i in range(vgg_num):
    model_list.append(vgg)

for i in range(wide_num):
    model_list.append(wide)

for i in range(squeeze_num):
    model_list.append(squeeze)

for i in range(mobile_num):
    model_list.append(mobile)

for i in range(mnas_num):
    model_list.append(mnas)

for i in range(inception_num):
    inception_list.append(inception)

for i in range(shuffle_num):
    model_list.append(shuffle)

for i in range(resx_num):
    model_list.append(resx)


# thread_list = []
# print('start')
# start = time.time()

# for model in model_list:
#     my_thread = Thread(target=predict, args=(model, inputs))
#     # target은 predict함수를 실행하고 return값(output값)을 target에 저장
#     # args는 predict함수에 전달되는 2개의 parameter
#     my_thread.start()
#     thread_list.append(my_thread)

# for model in inception_list:
#     my_thread = Thread(target=predict, args=(model, inputs2))
#     my_thread.start()
#     thread_list.append(my_thread)

# for th in thread_list:
#     th.join()   # thread 종료

# torch.cuda.synchronize()
# end = time.time()
# print('pytorch-ori MULTI-THREAD',end-start)




# thread_list = []
# start_total = time.time()

# for model in model_list:
#     fut = torch.jit.fork(predict, model, inputs) # jit 컴파일러가 최적화를 해줌
#     thread_list.append(fut)

# for model in inception_list:
#     fut = torch.jit.fork(predict, model, inputs2)
#     thread_list.append(fut)

# torch.cuda.synchronize()
# end_total = time.time()
# print('pytorch-ori. SERIAL',end_total-start_total)


print('start')
start = torch.cuda.Event(enable_timing = True)
end = torch.cuda.Event(enable_timing = True)
start.record()
for model in model_list:
    predict(model,inputs) 
for model in inception_list:
    predict(model,inputs2)
end.record()    
torch.cuda.synchronize()

print('pytorch-ori exe time : ', start.elapsed_time(end)/1000,"'s")
