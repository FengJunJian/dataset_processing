
ksize = [3,3,2,2,3]
stride = [1,1,2,2,2]

def receptive_field(i):
    RF = 1
    for j in range(i, 0, -1):
        RF = (RF-1)*stride[j-1]+ksize[j-1]
    return RF

print(receptive_field(5)) # output=11