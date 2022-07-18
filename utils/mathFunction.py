import torch
def decop_ce(predicts,targets,lamb=2.0):
    i=0
    Batch,classNum=predicts.shape
    claSets=torch.arange(0,classNum).repeat([Batch, 1])
    tars=targets.view(-1,1).repeat([1, classNum])
    _,Cindx=torch.where(claSets==tars)
    _, CXindx = torch.where(claSets != tars)
    CXindx=CXindx.view(Batch,-1)
    results=torch.empty((0,),dtype=torch.float)
    for i,cls in enumerate(targets):
        results=torch.cat([results,
                           (-predicts[i][cls] + lamb*torch.sum(torch.log(torch.exp(predicts[i][CXindx[i]])))).view(1)],
                          dim=0)
    return results

x1=torch.randn((2,4),dtype=torch.float)
x=torch.tensor([[ 4.0, 1.0,  1.0,  1.0, 1.0],
                 [ 4.0,  1.0, 1.0, 1.0, 1.0]],dtype=torch.float)
# [[ 2.9421, -1.2860,  0.6242,  1.4711, -0.2317],
#         [ 1.3346,  0.6673, -1.6962, -1.3774, -0.2761]]
#vx=torch.max(x1,dim=1)[0]*2
#x=torch.cat([vx.view(-1,1),x1],dim=1)
label=torch.zeros(x.shape[0],dtype=torch.long)
predicts=x
targets=label
print(decop_ce(predicts,targets,1))

# torch.nn.CrossEntropyLoss
torch.nn.KLDivLoss


    # for i,cls in enumerate(targets):
    #     claSet=list(range(0,classNum))
    #     claSet.remove(targets[i])
    #     claSet=torch.tensor(claSet,dtype=torch.long)
    #     print(-predicts[i][targets[i]]+torch.sum(torch.log(torch.exp(predicts[i][claSet]))))
