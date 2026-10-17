import os


def write_label(root, image_rel, lines, lib="pose"):
    from autolabel.dataset import label_path_for

    path = label_path_for(root, image_rel, lib)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(line + "\n" for line in lines))
    return path


def idx_names(root):
    from autolabel.dataset import load_index

    return load_index(root).items.keys()
