# MG-Turbulent-Flow

## [DataSet](https://drive.google.com/drive/folders/1VOtLjfAkCWJePiacoDxC-nrgCREKvrpE?usp=sharing.)

Run the following command to create dataset for our model:

```
>>> mkdir rbc_data
>>> python3 ./data_gen.py
```

## Training

Run the following command to start the training session:

```
>>> python3 ./main.py \\
     --checkpoint_dir checkpoints \\
     --batch_size 32 \\
     --regulizer 1
```

| Variable             | Description                                                |
| -------------------- | ---------------------------------------------------------- |
| ```checkpoint_dir``` | Path to save the checkpoint of the model's params          |
| ```batch_size```     | Training and evaluating ```batch_size```                   |
| ```regulizer```      | Apply regulizer (**div W = 0**) when the value is non-zero |

### Velocity U & V Ground Truth and Prediction
![](Compare.gif)

### Reference
- [TF-Net](https://github.com/Rose-STL-Lab/Turbulent-Flow-Net)
