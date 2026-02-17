import torch

from cs336_basics.train.optimizer import SGD

for lr in [1e-1, 1e-2, 1e-3]:

    print("lr:", lr)
    # set seed for reproducibility before each experiment
    torch.manual_seed(0)
    cur_weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([cur_weights], lr=lr)

    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (cur_weights**2).mean() # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward() # Run backward pass, which computes gradients.
        opt.step() # Run optimizer step.
    
    print("end of training loop")

# lr: 0.1
# 26.271400451660156
# 26.166425704956055
# 26.092466354370117
# 26.032245635986328
# 25.980205535888672
# 25.93375015258789
# 25.891420364379883
# 25.85228729248047
# 25.81574249267578
# 25.781330108642578
# end of training loop
# lr: 0.01
# 26.271400451660156
# 26.260896682739258
# 26.25347137451172
# 26.247407913208008
# 26.242155075073242
# 26.237462997436523
# 26.23318099975586
# 26.229211807250977
# 26.225502014160156
# 26.222007751464844
# end of training loop
# lr: 0.001
# 26.271400451660156
# 26.270353317260742
# 26.269611358642578
# 26.269004821777344
# 26.268482208251953
# 26.26801109313965
# 26.267580032348633
# 26.267183303833008
# 26.266809463500977
# 26.266462326049805
# end of training loop