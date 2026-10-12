"""Class-centering as structured gradient perturbation, emulated CORRECTLY.

The previous version had a self-referential loss.  This version does what B actually
experiences: A returns gradientB[i] - per_class_mean(y_i), and B back-propagates THAT signal
to update its own bottom model.  A's head then trains on B's frozen embedding (main task),
and an attacker trains a separate head on the same frozen embedding with a small auxiliary
set of true labels (label inference).

    <python> scripts/probe_class_centering.py
"""

import torch

N, WIDTH, EMB, CLASSES = 4000, 32, 64, 2
SPLIT = 3000
AUX = 40


def trainBottom(features, truth, perturb):
    """Train B's bottom model from the gradient A returns (optionally class-centered)."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(WIDTH, 128), torch.nn.ReLU(), torch.nn.Linear(128, EMB))
    head = torch.nn.Linear(EMB, CLASSES)  # A's head, co-trained to produce the gradient
    optimizer = torch.optim.Adam(list(model.parameters()) + list(head.parameters()), lr=1e-3)

    for _ in range(1200):
        optimizer.zero_grad()
        logits = head(model(features))
        probs = torch.softmax(logits, dim=1)
        onehot = torch.nn.functional.one_hot(truth, CLASSES).float()
        grad = probs - onehot  # [N, C], the per-sample gradient A computes

        if perturb == "center":
            signal = grad.clone()
            for c in range(CLASSES):
                m = (truth == c)
                signal[m] = grad[m] - grad[m].mean(dim=0, keepdim=True)
        elif perturb == "gaussian":
            rms = grad.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-12)
            signal = grad + 0.5 * rms * torch.randn(grad.shape)
        else:
            signal = grad

        # B back-propagates the RETURNED signal (this is the actual VFL mechanic).
        logits.backward(signal)
        optimizer.step()

    with torch.no_grad():
        return model(features).detach()


def fitHead(embedding, truth, auxOnly=False):
    """A's head or the attacker's head: linear on the frozen embedding."""
    torch.manual_seed(1)
    head = torch.nn.Linear(embedding.shape[1], CLASSES)
    optimizer = torch.optim.Adam(head.parameters(), lr=5e-3)
    if auxOnly:
        xs, ys = embedding[:AUX], truth[:AUX]
    else:
        xs, ys = embedding[:SPLIT], truth[:SPLIT]
    for _ in range(600):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(xs), ys)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = head(embedding[SPLIT:]).argmax(1)
    return float((pred == truth[SPLIT:]).float().mean())


def main():
    torch.manual_seed(0)
    features = torch.randn(N, WIDTH)
    truth = (features @ torch.randn(WIDTH, CLASSES)).argmax(dim=1)

    print("  %-16s %12s %14s" % ("扰动", "主任务精度", "攻击者恢复率"))
    print("  " + "-" * 44)

    for perturb in ("none", "gaussian", "center"):
        embedding = trainBottom(features, truth, perturb)
        aAcc = fitHead(embedding, truth, auxOnly=False)      # A trains on full labels
        att = fitHead(embedding, truth, auxOnly=True)        # attacker trains on AUX labels
        print("  %-16s %12.4f %14.4f" % (perturb, aAcc, att))

    print()
    print("  随机水平：攻击者恢复率 = 0.5（二分类）。")


if __name__ == "__main__":
    main()
