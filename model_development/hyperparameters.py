from data.dataset import CustomDataset
import json
import yaml
import os

STD_MODEL_FOLDER = '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/model_development/models'
REQUIRED_TRAIN_PARAMS = ["model_name", "model_path", "architecture", "epochs", "batch_size", "img_size", "lr", "greyscale", "model_size", "patience", "training_data", "pretrained_model_path", "annotations_path"]
REQUIRED_EVAL_PARAMS = ["conf_threshold", "eval_data", "iou_association_threshold", "tracker"]
DYNAMIC_PARAMS = ["pretrained", "fine_tuning"]

def load_trained_models():
    with open("./assets/trained_models.json", "r") as file:
        return json.load(file)

def check_required_params(params, required_params):
    return all(param in params for param in required_params)

def create_model_path(model_name):
    model_path = f"{STD_MODEL_FOLDER}/{model_name}"
    assert not os.path.exists(model_path), f"Model {model_name} already exists."
    return model_path

def get_pretrained_model_path(model_name, trained_models):
    assert model_name in trained_models, "Pretrained model is not available."
    model_path = trained_models[model_name]["model_path"]
    assert os.path.exists(model_path), f"Pretrained model {model_name} does not exist."
    return model_path

def setup_dataset_params(training_data):
    """
    Checks if dataset_name is in Dataset.experimentation_dataset_path,
    if so, reads from the data_config.yaml file data_split, returning a dictionary 
    and data_autmentations, returning a list
    """
    assert "dataset_name" in training_data, "Dataset name is missing."

    dataset_params = {
        "dataset_name": training_data["dataset_name"],
    }

    if training_data["dataset_name"] in os.listdir(CustomDataset.experimentation_dataset_path):
        print(f"Using prebuilt dataset {training_data['dataset_name']}")
        data_config_path = os.path.join(CustomDataset.experimentation_dataset_path, training_data["dataset_name"], "data_config.yaml")
        with open(data_config_path, "r") as file:
            data_config = yaml.safe_load(file)

            dataset_params["datasets"] = data_config["data_split"]
            dataset_params["augmentations"] = data_config.get("data_augmentations", [])
            dataset_params["prebuilt"] = True
    else:
        print(f"No pre-existing dataset named {training_data['dataset_name']}")
        dataset_params["datasets"] = training_data["datasets"]
        dataset_params["augmentations"] = training_data["augmentations"]
        dataset_params["prebuilt"] = False
    
    return dataset_params

def construct_hyperparameters(**config):
    trained_models = load_trained_models()
    assert trained_models is not None, "Trained models data is not available."

    hyperparameters = {"model_name": config["model_name"]}
    model_pretrained = hyperparameters["model_name"] in trained_models
    fine_tuning = not model_pretrained and config["pretrained_model_path"] is not None

    if model_pretrained:
        assert config.get("pretrained_model_path", None) is None, "Pretrained model path is not required."
        model_train_params = trained_models[hyperparameters["model_name"]]
        assert model_train_params is not None and check_required_params(model_train_params, REQUIRED_TRAIN_PARAMS), "Missing required training parameters."
        hyperparameters.update({param: model_train_params[param] for param in REQUIRED_TRAIN_PARAMS})
        hyperparameters.update({param: config[param] for param in REQUIRED_EVAL_PARAMS})
    else:
        model_path = create_model_path(hyperparameters["model_name"])
        hyperparameters.update({**config, "model_path": model_path})
        if fine_tuning:
            assert "pretrained_model_path" in config, "Pretrained model path is missing."
            if '/' not in config["pretrained_model_path"]:
                # passed model name, not path
                model_path = get_pretrained_model_path(config["pretrained_model_path"], trained_models)
                hyperparameters["pretrained_model_path"] = model_path
        hyperparameters['annotations_path'] = os.path.join(hyperparameters["model_path"], "annotations.csv")

    hyperparameters["training_data"] = setup_dataset_params(config["training_data"])
    hyperparameters['pretrained'] = model_pretrained
    hyperparameters['fine_tuning'] = fine_tuning

    assert check_required_params(hyperparameters, REQUIRED_TRAIN_PARAMS + REQUIRED_EVAL_PARAMS + DYNAMIC_PARAMS), "Some required hyperparameters are missing."

    return hyperparameters

def save_hyperparameters(hyperparameters):
    with open("./assets/trained_models.json", "r+") as file:
        trained_models = json.load(file)
        trained_models[hyperparameters['model_name']] = hyperparameters
        file.seek(0)
        json.dump(trained_models, file, indent=4)
        file.truncate()