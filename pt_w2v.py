import time
import numpy as np
import torch
import torch.nn as nn

class w2w(nn.Module):
    def __init__(self, vocab_size, dim, nneg, batch_size, device):
        super(w2w, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, dim, sparse=True)
        self.output_emb = nn.Embedding(vocab_size, dim , sparse=True)
        self.loss_fct = torch.nn.BCEWithLogitsLoss( reduction="sum")
        self.t_target=torch.cat(
            [torch.ones([batch_size, 1],dtype=torch.float32, device=device),
            -torch.ones([batch_size, nneg], dtype=torch.float32, device=device)],dim=1 )



    def make_optimizer(self, ):
        self.optimizer = torch.optim.Adagrad(self.parameters())

    def forward(self, t_input_ids, t_output_ids):
        # first column of output_ids is true value
        t_X = self.input_emb(t_input_ids)
        t_Y = self.output_emb(t_output_ids)
        t_logits=(t_X.unsqueeze(1)*t_Y).sum(dim=2)
        t_loss=self.loss_fct(t_logits, self.t_target)
        return t_loss

vocab_size=987000
nneg=25
batch_size=256*1024
dim=128

#device="cpu"
device = "cuda:0" if torch.cuda.is_available()  else "cpu"
print("Using device", device)

model=w2w(vocab_size, dim, nneg, batch_size, device)
model.to(device)
optimizer=torch.optim.SGD(model.parameters(),lr=0.001)
rnd=np.random.RandomState(230718)

T=1000000//batch_size
start=time.time()
for t in range(T):
    if True:# t==0:
      input_ids=rnd.randint(low=0, high=vocab_size, size=[batch_size])
      output_ids=rnd.randint(low=0, high=vocab_size, size=[batch_size, nneg+1])
    model.zero_grad()
    loss=model(torch.tensor(input_ids, device=device), torch.tensor(output_ids, device=device))
    loss.backward()
    optimizer.step()

elapsed=time.time()-start
print("Batch size", batch_size, "Total", elapsed, "sec,", T/elapsed, "batch/sec", (T*batch_size)/elapsed, "pairs/sec")
