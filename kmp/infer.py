"""Stage III: infer from node features and optional precomputed PE, without edges."""
import argparse
import torch
from .models import Student
from .train import student_inference


def load_student(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    c = checkpoint["config"]
    model = Student(c["input_dim"], c["hidden_dim"], c["classes"], c["layers"],
                    c["student_dropout"], c["pe_dim"], c["pe_fusion"], c["batch_norm"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), c


def main():
    p = argparse.ArgumentParser(description="Graph-free student inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--features", required=True, help="Tensor saved using torch.save")
    p.add_argument("--pe")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = p.parse_args()
    model, config = load_student(args.checkpoint, args.device)
    features = torch.load(args.features, map_location="cpu", weights_only=True)
    pe = torch.load(args.pe, map_location="cpu", weights_only=True) if args.pe else None
    logits = student_inference(model, features, pe, config["batch_size"], args.device)
    torch.save(dict(logits=logits, predictions=logits.argmax(1)), args.output)


if __name__ == "__main__":
    main()
