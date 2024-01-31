# Considerations
### Yolo model
It requires a data path, where we have stored all the data in the yolo format. 
However, we want to be able to experiment with different data formats, so we have to create a
data loader that can read different formats and convert them to the yolo format.

\\
To use it with the Yolo model, we had to create a new Trainer class, that extends the original `DetectionTrainer`.
One problem with this is that the `DetectionTrainer` checks the data format in the `init` method, so we had to
comment out those lines. This is not ideal, but it works for now.
\\ 
To replicate the behaviour, comment lines 129-139 of `Ultralytics/engine/trainer.py` 
