"""Experimentos de la Clase 07 con Neurito.
Ejecutar desde la raíz del proyecto:
    uv run --with matplotlib --with pillow python experimentos/experimentos.py
Genera las imágenes en capturas/ y los números en experimentos/resultados.json
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import neurito_lab as lab

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "capturas"
OUT.mkdir(exist_ok=True)
R = {}
def _font(candidates, size=17):
    for c in candidates:
        try:
            return ImageFont.truetype(c, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


MONO = _font(["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "C:/Windows/Fonts/consola.ttf", "DejaVuSansMono.ttf"])
MONO_B = _font(["/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", "C:/Windows/Fonts/consolab.ttf", "DejaVuSansMono-Bold.ttf"])
NAMES = {"eat": "EAT (comer)", "sleep": "SLEEP (dormir)", "play": "PLAY (jugar)"}


CMD = "uv run --with matplotlib --with pillow python experimentos/experimentos.py"


def terminal(lines, path, title="Terminal", cmd="uv run main.py"):
    lines = [f"PS C:\\Users\\crond\\Documents\\neurito> {cmd}"] + lines
    w = max(760, 24 + 10.3 * max(len(l) for l in lines))
    h = 60 + 24 * len(lines) + 16
    img = Image.new("RGB", (int(w), h), (30, 30, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 32], fill=(45, 45, 48))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([12 + i * 20, 10, 24 + i * 20, 22], fill=c)
    d.text((90, 7), title, font=MONO, fill=(200, 200, 200))
    for i, l in enumerate(lines):
        col = (86, 182, 194) if i == 0 else (212, 212, 212)
        if "accuracy" in l.lower() or "precisión" in l.lower() or l.startswith(">>"):
            col = (152, 195, 121)
        d.text((14, 46 + i * 24), l, font=MONO_B if i == 0 else MONO, fill=col)
    img.save(path)


def plot_style():
    plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .3,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


plot_style()
COL = {"eat": "#EB7D4B", "sleep": "#46AF73", "play": "#697DE1"}

# ---------------- Base ----------------
lines = []
base = lab.train(verbose_lines=lines)
P = base["params"]
terminal(lines, OUT / "01_entrenamiento_base_6_neuronas.png", "Neurito - entrenamiento base (3-6-3, 1200 épocas, lr 0.1)")
R["base"] = dict(acc=float(base["acc"]), final_loss=float(base["losses"][-1]))
lab.render(P, [0.2, 0.9, 0.2], OUT / "02_ventana_neurito_estado_calma.png")

# ---------------- Exp 1: sweeps ----------------
sweeps = {
    "hambre": dict(idx=0, fixed=[None, 0.5, 0.3], theory="hambre > 0.50 (cansancio fijo 0.50)"),
    "energia": dict(idx=1, fixed=[0.5, None, 0.3], theory="energía < 0.50 (cansancio > 0.50 = hambre fija)"),
    "aburrimiento": dict(idx=2, fixed=[0.5, 0.6, None], theory="aburrimiento > 0.50 (hambre fija 0.50)"),
}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
R["exp1"] = {}
grid = np.round(np.linspace(0, 1, 1001), 3)
for ax, (name, s) in zip(axes, sweeps.items()):
    data = np.array([[g if v is None else v for v in s["fixed"]] for g in grid])
    probs = lab.predict(P, data)
    act = np.argmax(probs, 1)
    changes = [(float(grid[i]), lab.actions[act[i - 1]], lab.actions[act[i]]) for i in range(1, len(grid)) if act[i] != act[i - 1]]
    for k, a in enumerate(lab.actions):
        ax.plot(grid, probs[:, k], color=COL[a], lw=2, label=NAMES[a])
    for c in changes:
        ax.axvline(c[0], color="k", ls="--", lw=1)
        ax.text(c[0] + .02, .92, f"cambio en {c[0]:.2f}", fontsize=8)
    fixed_txt = ", ".join(f"{n}={v}" for n, v in zip(["hambre", "energía", "aburr."], s["fixed"]) if v is not None)
    ax.set_title(f"Variando {name.replace('energia','energía')}\n({fixed_txt})", fontsize=10)
    ax.set_xlabel(f"{name.replace('energia','energía')} (0 a 1)"); ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilidad Softmax")
    R["exp1"][name] = dict(fixed=s["fixed"], changes=changes, theory=s["theory"])
    # screenshots each side of the first change
    c0 = changes[0][0]
    for tag, val in (("antes", round(c0 - 0.1, 1)), ("despues", round(c0 + 0.1, 1))):
        needs = [val if v is None else v for v in s["fixed"]]
        p, st = lab.render(P, needs, OUT / f"03_exp1_{name}_{tag}_del_cambio.png")
        R["exp1"][name][tag] = dict(needs=needs, probs=[round(float(x) * 100, 1) for x in p], state=st)
axes[0].legend(loc="center left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "04_exp1_grafica_barrido_necesidades.png"); plt.close(fig)

# ---------------- Exp 2: near ties ----------------
cases = [
    ("Prioridad clara (hambre alta)", [0.90, 0.80, 0.10]),
    ("Empate exacto 0.50 / 0.50 / 0.50", [0.50, 0.50, 0.50]),
    ("Casi empate: hambre 0.52", [0.52, 0.50, 0.50]),
    ("Casi empate: cansancio 0.52", [0.50, 0.48, 0.50]),
    ("Casi empate: aburrimiento 0.52", [0.50, 0.50, 0.52]),
    ("Casi empate alto 0.80/0.79/0.78", [0.80, 0.21, 0.78]),
]
R["exp2"] = []
tl = []
for label, needs in cases:
    p = lab.predict(P, np.array([needs]))[0]
    pol = lab.actions[int(lab.policy(np.array([needs]))[0])] if len(set([needs[0], 1 - needs[1], needs[2]])) == 3 else "empate"
    R["exp2"].append(dict(label=label, needs=needs, probs=[round(float(x) * 100, 1) for x in p],
                          pred=lab.actions[int(np.argmax(p))], policy=pol))
    tl.append(f"{label:<34} h={needs[0]:.2f} e={needs[1]:.2f} a={needs[2]:.2f} -> "
              f"EAT {p[0]*100:5.1f}%  SLEEP {p[1]*100:5.1f}%  PLAY {p[2]*100:5.1f}%")
terminal(tl, OUT / "05_exp2_softmax_prioridades_casi_identicas.png", "Neurito - Softmax con prioridades casi idénticas",
         cmd=CMD)
lab.render(P, [0.90, 0.80, 0.10], OUT / "06_exp2_ventana_prioridad_clara.png")
lab.render(P, [0.80, 0.21, 0.78], OUT / "07_exp2_ventana_casi_empate.png")

# ---------------- Exp 3: hidden neurons ----------------
R["exp3"] = []
fig, ax = plt.subplots(figsize=(7.5, 4))
tl = []
for hdn in (3, 6, 10, 20):
    r = lab.train(hidden=hdn)
    R["exp3"].append(dict(hidden=hdn, loss=float(r["losses"][-1]), acc=float(r["acc"])))
    ax.plot(r["losses"], lw=2, label=f"{hdn} neuronas ocultas")
    tl.append(f"3-{hdn:<2}-3 | pérdida final: {r['losses'][-1]:.4f} | precisión prueba: {r['acc']:.2%}")
ax.set_xlabel("Época"); ax.set_ylabel("Pérdida (entropía cruzada)"); ax.set_title("Pérdida según neuronas ocultas (1200 épocas, lr 0.1)")
ax.legend(); fig.tight_layout(); fig.savefig(OUT / "09_exp3_grafica_perdida_neuronas_ocultas.png"); plt.close(fig)
terminal(tl, OUT / "08_exp3_resultados_neuronas_ocultas.png", "Neurito - comparación de neuronas ocultas",
         cmd=CMD)

# ---------------- Exp 4: epochs & learning rate ----------------
R["exp4_epochs"] = []; R["exp4_lr"] = []
fig, axs = plt.subplots(1, 2, figsize=(13, 4))
tl = []
for ep in (100, 300, 1200, 3000, 6000):
    r = lab.train(epochs=ep)
    R["exp4_epochs"].append(dict(epochs=ep, loss=float(r["losses"][-1]), acc=float(r["acc"])))
    tl.append(f"épocas={ep:<5} lr=0.1  | pérdida final: {r['losses'][-1]:.4f} | precisión: {r['acc']:.2%}")
    if ep == 6000:
        axs[0].plot(r["losses"], lw=2, color="#444")
        for m in (100, 300, 1200, 3000):
            axs[0].plot(m - 1, r["losses"][m - 1], "o", ms=6)
            axs[0].annotate(f"{m} ép.\n{r['losses'][m-1]:.3f}", (m - 1, r["losses"][m - 1]), textcoords="offset points", xytext=(6, 8), fontsize=8)
axs[0].set_title("Más épocas (lr 0.1)"); axs[0].set_xlabel("Época"); axs[0].set_ylabel("Pérdida")
tl.append("")
for lr in (0.01, 0.1, 0.5, 1.0, 5.0, 20.0):
    r = lab.train(lr=lr)
    L = r["losses"]
    osc = float(np.mean(np.abs(np.diff(L[-200:]))))
    R["exp4_lr"].append(dict(lr=lr, loss=float(L[-1]), acc=float(r["acc"]), osc=osc, maxloss=float(np.nanmax(L))))
    axs[1].plot(L, lw=1.6, label=f"lr = {lr}")
    tl.append(f"épocas=1200  lr={lr:<5}| pérdida final: {L[-1]:.4f} | precisión: {r['acc']:.2%} | oscilación: {osc:.4f}")
axs[1].set_yscale("log"); axs[1].set_title("Tasa de aprendizaje (1200 épocas)"); axs[1].set_xlabel("Época"); axs[1].set_ylabel("Pérdida (escala log)")
axs[1].legend(fontsize=8, loc="lower left", ncol=3)
fig.tight_layout(); fig.savefig(OUT / "11_exp4_grafica_epocas_y_tasa_aprendizaje.png"); plt.close(fig)
terminal(tl, OUT / "10_exp4_resultados_epocas_y_tasa_aprendizaje.png", "Neurito - épocas y tasa de aprendizaje",
         cmd=CMD)

# ---------------- Exp 5: fourth action ----------------
def policy4(d):
    return np.argmax(np.column_stack((d[:, 0], 1 - d[:, 1], d[:, 2], d[:, 3])), axis=1)
lines = []
r4 = lab.train(n_in=4, n_out=4, make_targets=policy4, verbose_lines=lines)
cnt = np.bincount(r4["targets"], minlength=4)
lines = ["Entradas: hambre, energía, aburrimiento, suciedad  ->  salidas: EAT, SLEEP, PLAY, BATHE", ""] + lines + [
    "", "Ejemplos por etiqueta (3000): " + ", ".join(f"{a}={c}" for a, c in zip(["EAT", "SLEEP", "PLAY", "BATHE"], cnt))]
terminal(lines, OUT / "12_exp5_prototipo_cuarta_accion_banarse.png", "Neurito - prototipo con 4 acciones (4-6-4)",
         cmd=CMD)
R["exp5"] = dict(acc=float(r4["acc"]), loss=float(r4["losses"][-1]), counts=cnt.tolist())

# ---------------- Q1: weights vs biases ----------------
np.set_printoptions(precision=3, suppress=True)
tl = ["W1 (3 entradas x 6 ocultas):"] + ["  " + str(row) for row in P["W1"]] + \
     ["b1 (1 x 6):", "  " + str(P["b1"][0]), "", "W2 (6 ocultas x 3 salidas):"] + ["  " + str(row) for row in P["W2"]] + \
     ["b2 (1 x 3):", "  " + str(P["b2"][0]),
      "", f"Total parámetros: pesos = {P['W1'].size + P['W2'].size}, sesgos = {P['b1'].size + P['b2'].size}"]
terminal(tl, OUT / "13_pregunta1_pesos_y_sesgos_entrenados.png", "Neurito - pesos y sesgos tras el entrenamiento",
         cmd=CMD)
R["params"] = {k: v.round(3).tolist() for k, v in P.items()}

# ---------------- Q4: misleading accuracy ----------------
test, tt = base["test"], base["test_targets"]
scores = np.column_stack((test[:, 0], 1 - test[:, 1], test[:, 2]))
srt = np.sort(scores, 1)
margin = srt[:, -1] - srt[:, -2]
pred = np.argmax(lab.predict(P, test), 1)
ok = pred == tt
q4 = {}
for lo, hi in ((0, .02), (.02, .05), (.05, .1), (.1, 1.01)):
    m = (margin >= lo) & (margin < hi)
    q4[f"{lo}-{hi}"] = dict(n=int(m.sum()), acc=float(ok[m].mean()))
# rule the network never learned: neutral state
calm = lab.predict(P, np.array([[0.2, 0.9, 0.2]]))[0]
# out of range input (needs are clipped to 0..1 in the game, but a buggy input)
oor = lab.predict(P, np.array([[1.5, 1.0, 1.4]]))[0]
cm = np.zeros((3, 3), int)
for a, b in zip(tt, pred):
    cm[a, b] += 1
tl = ["Matriz de confusión (filas = política real, columnas = predicción):",
      "            EAT   SLEEP  PLAY"] + [f"  {n:<8}{r[0]:>5}{r[1]:>7}{r[2]:>6}" for n, r in zip(["EAT", "SLEEP", "PLAY"], cm)] + [
      "", "Precisión según qué tan 'reñida' está la decisión (margen entre las 2 necesidades mayores):"] + [
      f"  margen {k:<10} -> {v['n']:>4} casos | precisión {v['acc']:.2%}" for k, v in q4.items()] + [
      "", f"Estado de calma (0.2, 0.9, 0.2): la red dice EAT {calm[0]*100:.1f}% SLEEP {calm[1]*100:.1f}% PLAY {calm[2]*100:.1f}%",
      "  -> 'CALM' no lo decide la red: es una regla fija del dibujo (if en el bucle)."]
terminal(tl, OUT / "14_pregunta4_precision_enganosa.png", "Neurito - ¿qué esconde el 99.4%?", cmd=CMD)
R["q4"] = dict(bins=q4, cm=cm.tolist(), calm=[float(x) for x in calm], oor=[float(x) for x in oor])

(Path(__file__).resolve().parent / "resultados.json").write_text(json.dumps(R, indent=1, ensure_ascii=False))
print(json.dumps(R, indent=1, ensure_ascii=False)[:6000])
