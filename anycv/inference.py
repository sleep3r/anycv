import fire
from mmengine import Config
from mmengine.registry import DATASETS, MODELS, DefaultScope
import numpy as np
import onnxruntime as ort


def build_dataset_and_transform(config: Config) -> tuple:
    """
    Build dataset and transformation pipeline from configuration.

    Args:
        config (Config): Configuration object.

    Returns:
        Tuple: Contains dataset, transform pipeline, and data preprocessor (if present).
    """
    test_dataset = DATASETS.build(config.test_dataloader.dataset)
    transform = test_dataset.pipeline
    model = MODELS.build(config.model)
    data_preprocessor = getattr(model, "data_preprocessor", None)
    return test_dataset, transform, data_preprocessor


def preprocess_image(image_path: str, transform, data_preprocessor=None) -> np.ndarray:
    """
    Preprocess image using the transformation pipeline and optionally a data preprocessor.

    Args:
        image_path (str): Path to the input image.
        transform: Transformation pipeline.
        data_preprocessor: Data preprocessor (if available).

    Returns:
        np.ndarray: Preprocessed image suitable for model input.
    """
    transformed_data = transform({"img_path": image_path})
    transformed_data.pop("data_samples")

    if data_preprocessor is not None:
        preprocessed_data = data_preprocessor(transformed_data)["inputs"]

    return preprocessed_data.unsqueeze(0).numpy()


def run_inference(
    ort_session: ort.InferenceSession, input_data: np.ndarray
) -> np.ndarray:
    """
    Run inference on the input data using the ONNX model.

    Args:
        ort_session (ort.InferenceSession): ONNX runtime inference session.
        input_data (np.ndarray): Preprocessed input data.

    Returns:
        np.ndarray: Model output.
    """
    input_name = ort_session.get_inputs()[0].name
    ort_outs = ort_session.run(None, {input_name: input_data})
    return ort_outs


def get_prediction(ort_outs: np.ndarray, dataset) -> tuple[str, float]:
    """
    Get prediction and confidence from the model output.

    Args:
        ort_outs (np.ndarray): Model output.
        dataset: Dataset object containing metadata.

    Returns:
        Tuple[str, float]: Predicted class and confidence score.
    """
    pred = np.argmax(ort_outs[0])
    conf = np.max(ort_outs[0])
    return dataset._metainfo["classes"][pred], conf


def main(project: str, config_path: str, model_path: str, image_path: str):
    """
    Main function to perform inference using ONNX model.

    Args:
        project (str): mmlab project name.
        config_path (str): Path to the configuration file.
        model_path (str): Path to the ONNX model file.
        image_path (str): Path to the input image.
    """
    config = Config.fromfile(config_path)
    with DefaultScope.overwrite_default_scope(scope_name=project):
        test_dataset, transform, data_preprocessor = build_dataset_and_transform(config)

    ort_session = ort.InferenceSession(model_path)
    input_data = preprocess_image(image_path, transform, data_preprocessor)
    ort_outs = run_inference(ort_session, input_data)
    pred, conf = get_prediction(ort_outs, test_dataset)
    print(f"Prediction: {pred}, Confidence: {conf}")


if __name__ == "__main__":
    fire.Fire(main)
