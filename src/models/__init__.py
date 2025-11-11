import importlib
import inspect
from pathlib import Path
from typing import Dict, List

def list_available_models() -> Dict[str, List[str]]:
    """
    List all available models organized by file.

    Returns:
        Dict mapping filename to list of class names
    """
    models_dir = Path(__file__).parent
    available = {}

    for file in models_dir.glob('*.py'):
        if file.stem in ['__init__', '__pycache__']:
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
    Format: 'filename.ClassName' or 'filename_ClassName'

    Args:
        model_name: 'filename.ClassName' or 'filename_ClassName'
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
        class_name = model_name.upper()

    try:
        # Import module
        module = importlib.import_module(f'.{file_name.lower()}', package='models')

        # Get the class
        if hasattr(module, class_name):
            model_class = getattr(module, class_name)
        else:
            available_classes = [name for name, obj in inspect.getmembers(module, inspect.isclass)
                               if obj.__module__ == module.__name__]
            raise ValueError(
                f"Class '{class_name}' not found in models/{file_name}.py. "
                f"Available classes: {available_classes}"
            )

        # Instantiate
        return model_class(**kwargs)

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
