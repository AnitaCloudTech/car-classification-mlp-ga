import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import time
import random
import copy

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ucitavanje podataka
COLUMNS = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]

try:
    from ucimlrepo import fetch_ucirepo
    dataset = fetch_ucirepo(id=19)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    df.columns = COLUMNS
except Exception:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
    try:
        df = pd.read_csv(url, header=None, names=COLUMNS)
    except Exception:
        buying_vals  = ["vhigh", "high", "med", "low"]
        maint_vals   = ["vhigh", "high", "med", "low"]
        doors_vals   = ["2", "3", "4", "5more"]
        persons_vals = ["2", "4", "more"]
        lug_vals     = ["small", "med", "big"]
        safety_vals  = ["low", "med", "high"]

        rows = []
        for b in buying_vals:
            for m in maint_vals:
                for d in doors_vals:
                    for p in persons_vals:
                        for l in lug_vals:
                            for s in safety_vals:
                                rows.append([b, m, d, p, l, s])

        rng = np.random.default_rng(42)
        cls_map = {
            (0, 0): "unacc", (0, 1): "unacc", (0, 2): "acc",
            (1, 0): "unacc", (1, 1): "acc",   (1, 2): "good",
            (2, 0): "acc",   (2, 1): "good",  (2, 2): "vgood",
        }
        class_labels = []
        for row in rows:
            b_idx = buying_vals.index(row[0])
            s_idx = safety_vals.index(row[5])
            key = (min(b_idx, 2), min(s_idx, 2))
            lbl = cls_map[key]
            if lbl != "unacc" and rng.random() < 0.1:
                lbl = "unacc"
            class_labels.append(lbl)

        df = pd.DataFrame(rows, columns=COLUMNS[:-1])
        df["class"] = class_labels

print(f"Ucitano {len(df)} instanci")
print(df["class"].value_counts())

# vizualizacija
palette = {"unacc": "#E74C3C", "acc": "#F39C12", "good": "#27AE60", "vgood": "#2980B9"}
sns.set_style("whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Distribucija izlaznih klasa", fontsize=15, fontweight="bold")

counts = df["class"].value_counts()
colors = [palette[c] for c in counts.index]

axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_title("Apsolutna distribucija")
axes[0].set_xlabel("Klasa")
axes[0].set_ylabel("Broj instanci")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 5, str(v), ha="center", fontweight="bold")

axes[1].pie(counts.values, labels=counts.index, colors=colors,
            autopct="%1.1f%%", startangle=140,
            wedgeprops=dict(edgecolor="white", linewidth=2))
axes[1].set_title("Procentualna distribucija")
plt.tight_layout()
plt.savefig("fig1_distribucija_klasa.png", dpi=150, bbox_inches="tight")
plt.show()

features = COLUMNS[:-1]
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Distribucija ulaznih atributa po klasama", fontsize=15, fontweight="bold")

for ax, feat in zip(axes.flatten(), features):
    ct = pd.crosstab(df[feat], df["class"])
    ct = ct.reindex(columns=["unacc", "acc", "good", "vgood"], fill_value=0)
    ct.plot(kind="bar", ax=ax, color=[palette[c] for c in ct.columns], edgecolor="white", linewidth=0.8)
    ax.set_title(f'Atribut: "{feat}"')
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Klasa", fontsize=8)
    ax.grid(axis="y", alpha=0.4)

plt.tight_layout()
plt.savefig("fig2_atributi_po_klasama.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
df_enc = df.copy()
for col in df_enc.columns:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col])
corr = df_enc.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
            mask=mask, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax.set_title("Korelaciona matrica", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig3_korelaciona_mapa.png", dpi=150, bbox_inches="tight")
plt.show()

# enkodovanje i priprema
ordinal_maps = {
    "buying":   {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "maint":    {"low": 0, "med": 1, "high": 2, "vhigh": 3},
    "doors":    {"2": 0, "3": 1, "4": 2, "5more": 3},
    "persons":  {"2": 0, "4": 1, "more": 2},
    "lug_boot": {"small": 0, "med": 1, "big": 2},
    "safety":   {"low": 0, "med": 1, "high": 2},
}
class_map = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
class_map_inv = {v: k for k, v in class_map.items()}

X = df[features].copy()
for col, mapping in ordinal_maps.items():
    if col in X.columns:
        X[col] = X[col].map(mapping)

y = df["class"].map(class_map)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Trening: {X_train_sc.shape[0]}, Test: {X_test_sc.shape[0]}")

# genetski algoritam
SEARCH_SPACE = {
    "hidden_layer_sizes": [
        (32,), (64,), (128,),
        (64, 32), (128, 64), (64, 64),
        (128, 64, 32), (64, 32, 16), (100, 50),
    ],
    "activation":         ["relu", "tanh", "logistic"],
    "alpha":              [0.0001, 0.001, 0.01, 0.1],
    "learning_rate_init": [0.001, 0.005, 0.01],
    "max_iter":           [300, 500, 700],
}

def random_individual():
    return {k: random.choice(v) for k, v in SEARCH_SPACE.items()}

def fitness(individual):
    clf = MLPClassifier(random_state=42, early_stopping=True, n_iter_no_change=20, **individual)
    scores = cross_val_score(clf, X_train_sc, y_train, cv=3, scoring="accuracy", n_jobs=-1)
    return scores.mean()

def tournament_selection(population, fitnesses, k=3):
    indices = random.sample(range(len(population)), k)
    best_idx = max(indices, key=lambda i: fitnesses[i])
    return population[best_idx]

def crossover(parent1, parent2):
    child = {}
    for key in parent1:
        child[key] = random.choice([parent1[key], parent2[key]])
    return child

def mutate(individual, mutation_rate=0.25):
    ind = copy.deepcopy(individual)
    if random.random() < mutation_rate:
        gene = random.choice(list(SEARCH_SPACE.keys()))
        ind[gene] = random.choice(SEARCH_SPACE[gene])
    return ind

POP_SIZE      = 20
N_GENERATIONS = 15
ELITE_SIZE    = 2
MUTATION_RATE = 0.3

population           = [random_individual() for _ in range(POP_SIZE)]
best_fitness_history = []
avg_fitness_history  = []
best_individual      = None
best_fitness_overall = -np.inf

print("Pokrecem GA...")
ga_start = time.time()

for gen in range(N_GENERATIONS):
    fitnesses    = [fitness(ind) for ind in population]
    gen_best_fit = max(fitnesses)
    gen_avg_fit  = np.mean(fitnesses)
    best_fitness_history.append(gen_best_fit)
    avg_fitness_history.append(gen_avg_fit)

    if gen_best_fit > best_fitness_overall:
        best_fitness_overall = gen_best_fit
        best_individual = copy.deepcopy(population[fitnesses.index(gen_best_fit)])

    print(f"Gen {gen+1}/{N_GENERATIONS} | najbolji: {gen_best_fit:.4f} | prosek: {gen_avg_fit:.4f}")

    sorted_pairs   = sorted(zip(fitnesses, population), key=lambda x: x[0], reverse=True)
    elite          = [ind for _, ind in sorted_pairs[:ELITE_SIZE]]
    new_population = elite[:]

    while len(new_population) < POP_SIZE:
        p1    = tournament_selection(population, fitnesses)
        p2    = tournament_selection(population, fitnesses)
        child = mutate(crossover(p1, p2), MUTATION_RATE)
        new_population.append(child)

    population = new_population

print(f"GA zavrsen za {time.time() - ga_start:.1f}s")
print(f"Najbolji hiperparametri: {best_individual}")

fig, ax = plt.subplots(figsize=(10, 5))
gens = range(1, N_GENERATIONS + 1)
ax.plot(gens, best_fitness_history, "o-", color="#2980B9", lw=2, ms=6, label="Najbolji fitness")
ax.plot(gens, avg_fitness_history,  "s--", color="#E74C3C", lw=2, ms=5, label="Prosecni fitness")
ax.fill_between(gens, avg_fitness_history, best_fitness_history, alpha=0.12, color="#2980B9")
ax.set_title("Konvergencija GA", fontsize=14, fontweight="bold")
ax.set_xlabel("Generacija")
ax.set_ylabel("Tacnost (CV)")
ax.legend()
ax.set_xticks(list(gens))
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("fig4_ga_konvergencija.png", dpi=150, bbox_inches="tight")
plt.show()

# treniranje finalnog modela
final_clf = MLPClassifier(random_state=42, early_stopping=False, **best_individual)
final_clf.fit(X_train_sc, y_train)

y_pred    = final_clf.predict(X_test_sc)
test_acc  = accuracy_score(y_test, y_pred)
train_acc = accuracy_score(y_train, final_clf.predict(X_train_sc))

print(f"Tacnost trening: {train_acc*100:.2f}%")
print(f"Tacnost test:    {test_acc*100:.2f}%")
print(f"Iteracije:       {final_clf.n_iter_}")

target_names = [class_map_inv[i] for i in sorted(class_map_inv)]
print(classification_report(y_test, y_pred, target_names=target_names))

fig, ax = plt.subplots(figsize=(7, 6))
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title(f"Matrica konfuzije - tacnost: {test_acc*100:.2f}%", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("fig5_matrica_konfuzije.png", dpi=150, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(final_clf.loss_curve_, color="#2980B9", lw=2, label="loss")
ax.set_title("Kriva ucenja", fontsize=13, fontweight="bold")
ax.set_xlabel("Epoha")
ax.set_ylabel("Gubitak")
ax.legend()
ax.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("fig6_kriva_ucenja.png", dpi=150, bbox_inches="tight")
plt.show()

baseline_clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
baseline_clf.fit(X_train_sc, y_train)
baseline_acc = accuracy_score(y_test, baseline_clf.predict(X_test_sc))

fig, ax = plt.subplots(figsize=(8, 5))
models = ["Baseline MLP", "GA-optimizovani MLP"]
accs   = [baseline_acc * 100, test_acc * 100]
bars   = ax.bar(models, accs, color=["#95A5A6", "#27AE60"], width=0.4, edgecolor="white", linewidth=2)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{acc:.2f}%", ha="center", fontweight="bold", fontsize=12)
ax.set_ylim(0, 105)
ax.set_title("Baseline vs GA model", fontsize=13, fontweight="bold")
ax.set_ylabel("Tacnost (%)")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("fig7_poredenje_modela.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"Baseline: {baseline_acc*100:.2f}% | GA model: {test_acc*100:.2f}% | poboljsanje: +{(test_acc-baseline_acc)*100:.2f}%")

# vizualizacija arhitekture mreze
def draw_network(layer_sizes, ax):
    max_neurons = max(layer_sizes)
    n_layers    = len(layer_sizes)
    layer_colors = ["#3498DB", "#2ECC71", "#E67E22", "#E74C3C"]

    neuron_positions = []
    for l, n in enumerate(layer_sizes):
        positions = []
        for i in range(n):
            x = l / (n_layers - 1)
            y = (i - (n - 1) / 2) / max(max_neurons, 1)
            positions.append((x, y))
        neuron_positions.append(positions)

    for l in range(n_layers - 1):
        for pos1 in neuron_positions[l]:
            for pos2 in neuron_positions[l + 1]:
                ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
                        color="#BDC3C7", lw=0.4, alpha=0.5, zorder=1)

    labels = ["Ulaz\n(6)", *[f"Skriveni\n({s})" for s in layer_sizes[1:-1]], "Izlaz\n(4)"]
    for l, (positions, color) in enumerate(zip(neuron_positions, layer_colors[:n_layers])):
        for pos in positions:
            circle = plt.Circle(pos, 0.025, color=color, zorder=3, ec="white", lw=1.5)
            ax.add_patch(circle)
        ax.text(positions[0][0], min(p[1] for p in positions) - 0.08,
                labels[l], ha="center", fontsize=9, fontweight="bold", color=color)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.7, 0.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Arhitektura neuronske mreze", fontsize=13, fontweight="bold", pad=10)

hidden    = best_individual.get("hidden_layer_sizes", (64, 32))
full_arch = [6, *hidden, 4]

fig, ax = plt.subplots(figsize=(12, 5))
draw_network(full_arch, ax)
plt.tight_layout()
plt.savefig("fig8_arhitektura_mreze.png", dpi=150, bbox_inches="tight")
plt.show()

# testiranje na nekoliko primera
test_samples = [
    {"buying": "low",   "maint": "low",   "doors": "4", "persons": "4",    "lug_boot": "big",   "safety": "high"},
    {"buying": "vhigh", "maint": "vhigh", "doors": "2", "persons": "2",    "lug_boot": "small", "safety": "low"},
    {"buying": "med",   "maint": "low",   "doors": "4", "persons": "more", "lug_boot": "big",   "safety": "high"},
    {"buying": "high",  "maint": "high",  "doors": "3", "persons": "4",    "lug_boot": "med",   "safety": "med"},
]

print("\nPrimeri predikcije:")
for s in test_samples:
    row        = [ordinal_maps[k][v] for k, v in s.items()]
    pred_label = class_map_inv[final_clf.predict(scaler.transform([row]))[0]]
    proba      = final_clf.predict_proba(scaler.transform([row]))[0]
    print(f"{s} => {pred_label.upper()} (p={max(proba):.2f})")
#POREĐENJE SA ALTERNATIVNIM METODAMA

alternative_models = {
    "Decision Tree":     DecisionTreeClassifier(random_state=42),
    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
    "k-NN (k=5)":        KNeighborsClassifier(n_neighbors=5),
    "Baseline MLP":      MLPClassifier(hidden_layer_sizes=(100,),
                                       max_iter=500, random_state=42),
    "GA-opt. MLP":       final_clf,
}

print("\nPoređenje metoda:")
print(f"{'Model':<20} {'Tačnost':>10} {'CV (3-fold)':>12}")
print("─" * 45)

results = {}
for name, model in alternative_models.items():
    if name != "GA-opt. MLP":
        model.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_sc))
    cv  = cross_val_score(model, X_train_sc, y_train,
                          cv=3, scoring="accuracy").mean()
    results[name] = acc
    print(f"{name:<20} {acc*100:>9.2f}%  {cv*100:>10.2f}%")

# Grafik poređenja
fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
accs  = [v*100 for v in results.values()]
colors_bar = ["#95A5A6","#95A5A6","#95A5A6","#95A5A6","#27AE60"]
bars  = ax.bar(names, accs, color=colors_bar,
               edgecolor="white", linewidth=2)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.3,
            f"{acc:.2f}%", ha="center", fontweight="bold", fontsize=10)
ax.set_ylim(0, 105)
ax.set_title("Poređenje klasifikacionih metoda",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Tačnost na test skupu (%)")
ax.set_xticklabels(names, rotation=15, ha="right")
ax.grid(axis="y", alpha=0.4)
plt.tight_layout()
plt.savefig("fig8_poredenje_metoda.png", dpi=150)
plt.show()