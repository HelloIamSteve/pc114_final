import torch
from pathlib import Path

from model import LeNet

def save_tensor_txt(tensor: torch.Tensor, path: Path) -> None:
    """Save tensor as flat float values, one value per line."""
    data = tensor.detach().cpu().contiguous().view(-1)
    with path.open("w", encoding="utf-8") as f:
        for value in data:
            f.write(f"{value.item():.10f}\n")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    weight_pt = script_dir / "LeNet_best.pt"
    output_dir = project_root / "weights"
    output_dir.mkdir(exist_ok=True)

    model = LeNet()
    state_dict = torch.load(weight_pt, map_location="cpu")
    model.load_state_dict(state_dict)

    sd = model.state_dict()

    mapping = {key: f'{key.replace(".", "_")}.txt' for key in sd.keys()}

    for key, filename in mapping.items():
        path = output_dir / filename
        save_tensor_txt(sd[key], path)
        print(f"saved {key:12s} shape={tuple(sd[key].shape)} -> {path.relative_to(project_root)}")

    print("Done. C++ can now load files from the weights/ folder.")

if __name__ == "__main__":
    main()