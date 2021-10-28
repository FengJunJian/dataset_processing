import wandb

wandb.init(project="demo",entity='chfjj',name='demo1')
wandb.config.dropout = 0.2
wandb.config.hidden_layer_size = 128
for i in range(10):
    loss=i
    wandb.log({'epoch':i,'loss':loss})