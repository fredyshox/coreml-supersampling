import os
import re
from urllib.parse import urlparse


def resolve_weights_uri(uri):
    comp = urlparse(uri)
    if len(comp.scheme) == 0:
        # local file
        return comp.path
    elif comp.scheme == "clearml":
        # clearml model output
        return _resolve_clearml_uri(comp.netloc, comp.path)
    else:
        raise ValueError(f"Unsupported weights uri: {uri}")  


def _resolve_clearml_uri(task_id, path):
    import clearml

    model_index = -1
    if len(path[1:]) != 0:
        model_index = int(path[1:])

    task = clearml.Task.get_task(task_id)
    try:
        model = task.get_models()["output"][model_index]
        model_path = model.get_local_copy()
        model_name = None
        for filename in os.listdir(model_path):
            prefix_match = re.match(r"^.*\.tf", filename)
            if prefix_match is not None:
                model_name = prefix_match[0]
                break
        
        if model_name is None:
            raise ValueError(f"Could not find tensorflow model in dir: {model_path}")

        return os.path.join(model_path, model_name)
    except IndexError:
        raise ValueError(f"Task {task_id} does not have model with index {model_index}")
