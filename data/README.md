# Files

## ADL_DCASE_DATA

This is the zip file of the DCASE 2016 dataset. Unzipped it has the following file format:
```bash
├── ADL_DCASE_DATA
│   ├── development
│   │   ├── audio/
│   │   └── labels.csv
│   └── evaluation
│       ├── audio/
│       └── labels.csv
```

Each `"audio/"` directory contains all of the data for that split. The data is stored as arrays where each array represents a spectrogram. The spectrograms have been created according to the parameters described in the paper. `"labels.csv"` contains the labels for each audio sample inside `"audio/"`. You should exclusively use the data in "development/" for training. The data in `"evaluation/"` is exclusively for evaluating your model. Do not train your model on the data in `"evaluation/"`. This data is also available directly on BlueCrystal at the following location: `/mnt/storage/scratch/qc19291/ADL_DATA/ADL_DCASE_DATA.zip`
Please make a copy of this data to your own scratch space before unzipping.


## dataset.py
This is a PyTorch Dataset implemention for the DCASE 2016 dataset. This code loads the spectrograms provided and splits them into shorter sequences. The DCASE class requires a path to your dataset and the length of your audio clips in seconds. You should use this class in conjuction with a PyTorch DataLoader. You can see examples of how to use a DataLoader in your lab code.

This dataset class will return tensors of the shape [batches, num_clips, height, width]. Most CNN models will expect data in the form [batches, channels, height, width]. In this case there is an additional dimension (num_clips) as a result of the sequence splitting described in the paper. In order to resolve this, we suggest you combine the number of clips into the batch dimension using torch.view(). You can then retrieve the correct dimensions by reshaping your data after passing it through the model. The DCASE class has a function, get_num_clips, which you can call. This function will return the number of clips each spectrogram is split into (determined by the clip length). You may find this useful when reshaping your tensors. We used a batch size of 64, although you can experiment with this. You do not need to worry about doing any time-shifting mentioned in the paper. Shuffling the data can be specified as an argument to your PyTorch DataLoader.

## VisualiseSpec.ipynb
This is a jupyter notebook that allows you to visualise a spectrogram. 






