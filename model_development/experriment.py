from architectures import yolov8
from trackers import botsort
from data import yolo_dataset, image_processor
from hyperparameters import construct_hyperparameters, save_hyperparameters
import wandb
import json
import os

architectures = {
  "yolov8": yolov8.YoloV8
}

trackers = {
  'botsort': botsort.BotSort
  }

dataset_mapping = {
 "yolov8": yolo_dataset.YoloDataset
  }


def prepare_dataset(hyperparameters, metrics):
    transforms = [image_processor.ImageProcessor.bgr2rgb]
    data_dir = "/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/images_raw/"
    data_config = hyperparameters['training_data']
    dataset_class = dataset_mapping[hyperparameters['architecture']]
    dataset = dataset_class(data_config['dataset_name'], data_dir, data_config['datasets'], data_config.get('augmentations', []), transforms)
    dataset.get_info()
    metrics['dataset_size'] = len(dataset)
    return dataset, metrics

def train_model(model, dataset, metrics):
    print("Training model...")
    train_time, dataset_time, device, final_model_path, mAP = model.train(dataset)
    metrics['training_time'] = train_time
    metrics['dataset_building_time'] = dataset_time
    metrics['training_device'] = device
    metrics['mAP'] = mAP
    return final_model_path, metrics

def evaluate_model(model, metrics):
    mota, motp, idf1, track_time, device = model.evaluate()
    metrics['mota'] = mota
    metrics['motp'] = motp
    metrics['idf1'] = idf1
    metrics['tracking_time'] = track_time
    metrics['tracking_device'] = device
    return metrics

def log_images(model_path):
    model_folder = os.path.dirname(model_path)
    plot_names = [f for f in os.listdir(model_folder) if os.path.isfile(os.path.join(model_folder, f)) and f.endswith('.png')]
    for plot in plot_names:
        wandb.log({plot: wandb.Image(os.path.join(model_folder, plot))})

def model_experimentation(hyperparameters):
    try:
        tracker = trackers[hyperparameters['tracker']]()
        model = architectures[hyperparameters['architecture']](hyperparameters, tracker)
        metrics = {}
        
        # Training
        if not hyperparameters['pretrained']:
            dataset, metrics = prepare_dataset(hyperparameters, metrics)
            metrics['dataset_size'] = len(dataset)
            final_model_path, metrics = train_model(model, dataset, metrics)
            hyperparameters['model_path'] = final_model_path

        metrics = evaluate_model(model, metrics)
        save_hyperparameters(hyperparameters)
        
        print('Initialising wandb...')
        wandb.init(project="SharkTrack", name=hyperparameters["model_name"], config=hyperparameters, job_type="training")
        log_images(hyperparameters['model_path'])
        wandb.log(metrics)
        wandb.finish()
    finally:
        wandb.finish()

def load_hyperparameters(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

if __name__ == "__main__":
    params_config = load_hyperparameters('./experiment.json')
    assert params_config is not None, "Hyperparameters are not available."

    hyperparameters = construct_hyperparameters(**params_config)
    print(hyperparameters)

    print("Starting experiment...")
    model_experimentation(hyperparameters)