import importlib
import inspect
from pathlib import Path
from typing import Dict, List

from data.dataset import MFCCDataset, SpectroDataset, WaveNetDataset, create_loaders

def list_available_models() -> Dict[str, List[str]]:
    """
    List all available models organized by file.

    Returns:
        Dict mapping filename to list of class names
    """
    models_dir = Path(__file__).parent
    available = {}

    for file in models_dir.glob('*.py'):
        if file.stem in ['__init__', '__pycache__', 'base']:
            continue

        try:
            module = importlib.import_module(f'.{file.stem}', package='models')
            classes = [name for name, obj in inspect.getmembers(module, inspect.isclass)
                      if obj.__module__ == module.__name__]
            if classes:
                available[file.stem] = classes
        except Exception:
            continue

    return available

def create_model(model_name: str, **kwargs):
    """
    Factory function to get model by name.
    Format: 'filename.ClassName' or 'filename_ClassName' or just 'filename'

    Args:
        model_name: 'filename.ClassName', 'filename_ClassName', or 'filename'
        **kwargs: Arguments to pass to the model constructor

    Returns:
        Instantiated model
    """
    # Parse the model name
    if '.' in model_name:
        file_name, class_name = model_name.split('.', 1)
    elif '_' in model_name:
        file_name, class_name = model_name.rsplit('_', 1)
    else:
        file_name = model_name.lower()
        class_name = None  # Will auto-select first available class

    try:
        # Import module
        module = importlib.import_module(f'.{file_name.lower()}', package='models')

        # Get available classes
        available_classes = [name for name, obj in inspect.getmembers(module, inspect.isclass)
                           if obj.__module__ == module.__name__]

        if not available_classes:
            raise ValueError(f"No classes found in models/{file_name}.py")


        if class_name is None:
            # Auto-select first available class
            class_name = available_classes[0]
            model_class = getattr(module, class_name)
        elif hasattr(module, class_name):
            model_class = getattr(module, class_name)
        else:
            raise ValueError(
                f"Class '{class_name}' not found in models/{file_name}.py. "
                f"Available classes: {available_classes}"
            )

        train_loader, test_loader, dataset, shape = create_loaders(
            dataset_class=model_class.supported_dataset(),
            batch_size=256,
            test_split=0.2,
            shuffle=True,
            random_seed=39,
            # does nothing on cpu:
            num_workers=40,
            pin_memory=True,
            persistent_workers=True
        )

        # set parameters
        if type(model_class.supported_dataset()) is SpectroDataset:
            kwargs.setdefault('input_channels', shape[0])
            kwargs.setdefault('H', shape[1])
            kwargs.setdefault('W', shape[2])
        elif type(model_class.supported_dataset()) is MFCCDataset:
            kwargs.setdefault('input_size', shape[1])

        kwargs.setdefault('num_classes', len(dataset.classes))

        # initiate model, with its loaders
        model = model_class(**kwargs)
        model.train_loader = train_loader
        model.test_loader = test_loader
        model.dataset = dataset
        model.data_shape = shape

        return model

    except ModuleNotFoundError:
        raise ValueError(
            f"Model file '{file_name}.py' not found.\n\n"
            f"Available models:\n{format_available_models()}"
        )
    except TypeError as e:
        raise ValueError(f"Error creating model '{model_name}': {e}")

def format_available_models() -> str:
    """Format available models as a readable string."""
    available = list_available_models()
    lines = []
    for file, classes in sorted(available.items()):
        for cls in classes:
            lines.append(f"  - {file}.{cls}")
    return "\n".join(lines) if lines else "  (none found)"
