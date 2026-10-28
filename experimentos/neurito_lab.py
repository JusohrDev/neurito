"""Helpers to reproduce Neurito's training/rendering with tunable hyperparameters."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
from pathlib import Path
import numpy as np
import pygame

BASE = Path(__file__).resolve().parent.parent  # carpeta raíz del proyecto (main.py, assets/)
src = (BASE / "main.py").read_text(encoding="utf-8")
ns = {}
exec(src.split("np.random.seed(42)")[0], ns)  # TEXT, LANGUAGE, labels, actions
TEXT, LANGUAGE = ns["TEXT"], ns["LANGUAGE"]
labels, actions = ns["labels"], ns["actions"]


def policy(data):
    return np.argmax(np.column_stack((data[:, 0], 1 - data[:, 1], data[:, 2])), axis=1)


def relu(v):
    return np.maximum(0, v)


def softmax(v):
    e = np.exp(v - np.max(v, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def train(hidden=6, epochs=1200, lr=0.1, seed=42, log_every=200, n_out=3,
          make_targets=None, n_in=3, verbose_lines=None):
    """Same steps and RNG order as main.py (seed -> X -> W1 -> W2 -> train -> test)."""
    make_targets = make_targets or policy
    np.random.seed(seed)
    X = np.random.rand(3000, n_in)
    targets = make_targets(X)
    W1 = np.random.randn(n_in, hidden) * 0.5
    b1 = np.zeros((1, hidden))
    W2 = np.random.randn(hidden, n_out) * 0.5
    b2 = np.zeros((1, n_out))
    losses = []
    for epoch in range(epochs):
        z1 = X @ W1 + b1
        h = relu(z1)
        p = softmax(h @ W2 + b2)
        n = len(X)
        loss = -np.mean(np.log(p[np.arange(n), targets] + 1e-10))
        losses.append(loss)
        d2 = p.copy(); d2[np.arange(n), targets] -= 1; d2 /= n
        gW2 = h.T @ d2; gb2 = d2.sum(0, keepdims=True)
        d1 = (d2 @ W2.T) * (z1 > 0)
        gW1 = X.T @ d1; gb1 = d1.sum(0, keepdims=True)
        W1 -= lr * gW1; b1 -= lr * gb1; W2 -= lr * gW2; b2 -= lr * gb2
        if verbose_lines is not None and log_every and epoch % log_every == 0:
            verbose_lines.append(f"{labels['epoch']} {epoch} | {labels['loss']}: {loss:.4f}")
    test = np.random.rand(1000, n_in)
    tt = make_targets(test)
    params = dict(W1=W1, b1=b1, W2=W2, b2=b2)
    pred = predict(params, test)
    acc = np.mean(np.argmax(pred, 1) == tt)
    if verbose_lines is not None:
        verbose_lines.append(f"{labels['accuracy']}: {acc:.2%}")
    return dict(params=params, losses=np.array(losses), acc=acc, test=test,
                test_targets=tt, X=X, targets=targets)


def predict(params, data):
    h = relu(data @ params["W1"] + params["b1"])
    return softmax(h @ params["W2"] + params["b2"])


# ---------- rendering (same drawing code as main.py) ----------
_screen = None
_frames = None


def _init():
    global _screen, _frames, font, small_font
    if _screen is not None:
        return
    pygame.init()
    _screen = pygame.display.set_mode((900, 600))
    font = pygame.font.SysFont(None, 30)
    small_font = pygame.font.SysFont(None, 23)
    asset_dir = BASE / "assets"
    pet_size = (330, 330)

    def load(names):
        return [pygame.transform.smoothscale(
            pygame.image.load(str(asset_dir / n)).convert_alpha(), pet_size) for n in names]
    _frames = {
        "neutral": load(["neutral_1.png", "neutral_2.png", "neutral_3.png"]),
        "eat": load(["comer_1.png", "comer_2.png"]),
        "sleep": load(["dormir_1.png", "dormir_2.png"]),
        "play": load(["jugar_1.png", "jugar_2.png"]),
    }


def render(params, needs, out_path, frame_index=0):
    _init()
    screen = _screen
    probabilities = predict(params, np.array([needs]))[0]
    action = actions[int(np.argmax(probabilities))]
    if needs[0] < 0.4 and needs[1] > 0.6 and needs[2] < 0.4:
        state = "neutral"
    else:
        state = action
    screen.fill((250, 245, 252))
    screen.blit(font.render(labels["title"], True, (55, 45, 75)), (45, 28))
    colors = [(235, 125, 75), (70, 175, 115), (105, 125, 225)]
    for i, key in enumerate(("hunger", "energy", "boredom")):
        v = needs[i]; y = 125 + i * 95
        screen.blit(small_font.render(f"{labels[key]}: {v * 100:.0f}%", True, (35, 40, 55)), (50, y - 27))
        pygame.draw.rect(screen, (215, 220, 230), (50, y, 290, 22), border_radius=7)
        pygame.draw.rect(screen, colors[i], (50, y, int(290 * v), 22), border_radius=7)
    fr = _frames[state]
    pet = fr[frame_index % len(fr)]
    screen.blit(pet, pet.get_rect(center=(650, 235)))
    screen.blit(font.render(labels[state], True, (65, 50, 85)), (585, 420))
    for i, a in enumerate(actions):
        screen.blit(small_font.render(f"{labels[a]}: {probabilities[i] * 100:.1f}%", True, (55, 50, 75)),
                    (50, 390 + i * 28))
    for j, k in enumerate(("controls1", "controls2", "controls3")):
        screen.blit(small_font.render(labels[k], True, (70, 70, 90)), (50, 495 + j * 30))
    pygame.image.save(screen, str(out_path))
    return probabilities, state
