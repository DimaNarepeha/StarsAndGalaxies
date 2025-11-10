import argparse, os, glob
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from Model import Model

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_images(inputs):
    files = []
    for p in inputs:
        if os.path.isdir(p):
            for root, _, fs in os.walk(p):
                for f in fs:
                    if f.lower().endswith(EXTS):
                        files.append(os.path.join(root, f))
        else:
            if p.lower().endswith(EXTS):
                files.append(p)
    return files


def infer_actual_class(path, classnames):
    """Return actual class name based on folder name if matches class list."""
    # Example: /.../data/test/galaxies/img123.jpg -> "galaxies"
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if p.lower() in [c.lower() for c in classnames]:
            return p
    return None  # unknown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--classes", nargs="+", default=["galaxies", "stars"])
    ap.add_argument("--size", type=int, default=128)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("[info] loading model...")
    model = Model.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        batch_size=64,
        learning_rate=1e-3,
        num_classes=len(args.classes),
        epsilon=1e-8
    ).to(device)
    model.eval()

    imgs = find_images(args.images)
    if not imgs:
        print("[error] no images found")
        return
    import random
    random.shuffle(imgs)
    for path in imgs:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[skip] {path}: {e}")
            continue

        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0]
        idx = int(prob.argmax())
        conf = float(prob[idx])
        pred = args.classes[idx]

        actual = infer_actual_class(path, args.classes)
        actual_str = actual if actual is not None else "?"

        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Actual: {actual_str}\nPredicted: {pred}\nConf: {conf:.3f}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
