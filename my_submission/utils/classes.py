DEFAULT_CLASSES = ["person", "car", "dog", "cat", "chair"]
CLASSES = list(DEFAULT_CLASSES)
CLASS_TO_IDX = {name: index for index, name in enumerate(CLASSES)}
IDX_TO_CLASS = {index: name for name, index in CLASS_TO_IDX.items()}


def set_active_classes(class_names: list[str]) -> None:
    if not class_names:
        raise ValueError("class_names must not be empty")
    if len(set(class_names)) != len(class_names):
        raise ValueError("class_names contains duplicates")
    CLASSES[:] = class_names
    CLASS_TO_IDX.clear()
    CLASS_TO_IDX.update({name: index for index, name in enumerate(CLASSES)})
    IDX_TO_CLASS.clear()
    IDX_TO_CLASS.update({index: name for name, index in CLASS_TO_IDX.items()})
