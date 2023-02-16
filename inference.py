import torch

from models import *

# set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(checkpoint_path, model, entries):
    loaded_checkpoint = torch.load(checkpoint_path, map_location = device)
    
    model = model.to(device)
    model.load_state_dict(loaded_checkpoint['best_model'])
    model.eval()
    
    inputs = entries.unsqueeze(0).to(device)
    preds = inputs[0]

    # start inference loops
    with torch.no_grad():
        for _ in range(50):
            output = model(inputs)
            inputs = torch.concat([inputs[:, 4:], output], dim = 1)
            
            preds = torch.concat([preds, output[0]], dim = 0)
    
    return preds

if __name__ == '__main__':
    checkpoint_path = 'checkpoints/FNO2d_rbc_data_16_4_checpoint.pth'
    model = FNO2d((2, 64, 64), 16, 4, width = 128)
    
    # set up the entries of the turbu~lent flow
    data = torch.load('data/sample_5.pt')
    
    preds = test(checkpoint_path, model, data[:16])
    preds = preds.data.cpu().numpy()
    
    data = data.numpy()
    
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    
    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].set_title('Ground truth x')
    axs[1, 0].set_title('Ground truth y')
    axs[0, 1].set_title('Prediction x')
    axs[1, 1].set_title('Prediction y')
    
    def animate(i):
        axs[0, 0].imshow(data[i][0])
        axs[1, 0].imshow(data[i][1])
        axs[0, 1].imshow(preds[i][0])
        axs[1, 1].imshow(preds[i][1])
        
    ani = FuncAnimation(fig, animate, frames = 100, interval = 50, repeat = False)
    plt.show()
