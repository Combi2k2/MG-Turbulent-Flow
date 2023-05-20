'''
    Modify the MODEL_NAME and EXPERIMENT_NAME
    Modify the CHECKPOINT_PATH to match the given experiment
    Modify the TEST_DATA_PATH
'''
import torch

from models import *

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(model, inputs):
    inputs = inputs.unsqueeze(0).to(device)
    preds = inputs[0]

    # start inference loops
    with torch.no_grad():
        for _ in range(50):
            output = model(inputs)
            inputs = torch.concat([inputs[:, 4:], output], dim = 1)
            
            preds = torch.concat([preds, output[0]], dim = 0)
    
    return preds

MODEL_NAME = 'MGxTransformer'
EXPERIMENT_NAME = 'rbc_data_16_4'
TEST_DATA_PATH = 'data/sample_5.pt'
CHECKPOINT_PATH = f'{MODEL_NAME}_{EXPERIMENT_NAME}_checkpoint.pth'

loaded_checkpoint = torch.load(CHECKPOINT_PATH, map_location = device)

model = MGxTransformer((2, 64, 64), 16, 4)
model = model.to(device)
model.load_state_dict(loaded_checkpoint['best_model'])
model.eval()
    
# set up the entries of the turbu~lent flow
data = torch.load(TEST_DATA_PATH)
preds = predict(model, data[:16])

torch.save(preds, f'demo/{MODEL_NAME}_{EXPERIMENT_NAME}.pt')
    
# data = data.numpy()

# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation

# fig, axs = plt.subplots(2, 2)

# axs[0, 0].set_title('Ground truth x')
# axs[1, 0].set_title('Ground truth y')
# axs[0, 1].set_title('Prediction x')
# axs[1, 1].set_title('Prediction y')

# def animate(i):
#     axs[0, 0].imshow(data[i][0])
#     axs[1, 0].imshow(data[i][1])
#     axs[0, 1].imshow(preds[i][0])
#     axs[1, 1].imshow(preds[i][1])
    
# ani = FuncAnimation(fig, animate, frames = 100, interval = 50, repeat = False)
# plt.show()
