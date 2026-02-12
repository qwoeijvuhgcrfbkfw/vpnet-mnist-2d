import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

def to_ax_frac_x(ax, x):
    lim = ax.get_xlim()
    return (x - lim[0]) / (lim[1] - lim[0])

def to_ax_frac_y(ax, y):
    lim = ax.get_ylim()
    return (y - lim[0]) / (lim[1] - lim[0])

df = pd.read_csv("../150_epochs_study_results.csv")
df = df[df["state"] == "COMPLETE"]
df["pcount"] = (((df["params_num_hermite_directional_coeffs"] ** 2) * df["params_hidden_layer_neuron_count"] +
                df["params_hidden_layer_neuron_count"] * (10 + 1) + 10 + 4) * 2).pow(0.5)
df["true_pcount"] = ((df["params_num_hermite_directional_coeffs"] ** 2) * df["params_hidden_layer_neuron_count"] +
                df["params_hidden_layer_neuron_count"] * (10 + 1) + 10 + 4)


plt.figure(figsize=[13, 10])

pivoted = df.pivot(index="params_num_hermite_directional_coeffs", columns="params_hidden_layer_neuron_count", values="value")
sns.heatmap(pivoted, annot=True, cmap="YlGnBu", fmt='.2f', center=95, vmin=95)

plt.gca().set_ylabel("Directional Hermite coefficients")
plt.gca().set_xlabel("Hidden layer neuron count")

plt.tight_layout()

plt.show()

plt.scatter(
    y=df["params_num_hermite_directional_coeffs"],
    x=df["params_hidden_layer_neuron_count"],
    c=df["value"],
    s=df["pcount"]
)

plt.gca().invert_yaxis()
plt.gca().set_ylabel("Directional Hermite coefficients")
plt.gca().set_xlabel("Hidden layer neuron count")

plt.show()

sns.scatterplot(
    data=df,
    x="true_pcount",
    y="value",
    hue="params_num_hermite_directional_coeffs",
    palette="tab10",
    alpha=0.7
)

for K, g in df.sort_values(["params_num_hermite_directional_coeffs", "true_pcount"]).groupby("params_num_hermite_directional_coeffs"):
    plt.gca().plot(g["true_pcount"].values, g["value"].values, linewidth=1.3, alpha=0.4)

anpt_idxs = df["value"].nlargest(5).index

for i, idx in enumerate(anpt_idxs):
    x = df["true_pcount"][idx]
    y = df["value"][idx]

    print(f"accuracy: {y}; pcount: {x}; hermite_coeffs: {df['params_num_hermite_directional_coeffs'][idx]}, hidden_layer_neurons: {df['params_hidden_layer_neuron_count'][idx]}, number: {i + 1}")

    if i + 1 != 5:
        if i + 1 == 1:
            plt.gca().annotate(f"{i + 1},5", (x, y),
                           xytext=(0, 3), textcoords="offset points", fontsize=12)
        elif i + 1 == 2:
            plt.gca().annotate(f"{i + 1}", (x, y),
                           xytext=(3, 3), textcoords="offset points", fontsize=12)
        else:
            plt.gca().annotate(f"{i + 1}", (x, y),
                           xytext=(0, 3), textcoords="offset points", fontsize=12)

        plt.gca().axvline(x, ymin=0, ymax=to_ax_frac_y(plt.gca(), y),
                   color='gray', linestyle='--', linewidth=0.8)

plt.gca().legend(title="Hermite coefficients, $K$",
          bbox_to_anchor=(0.98, 0.02),  # fraction of axes
          loc="lower right",
          ncol=3,
          title_fontsize=11,
          frameon=True)

plt.gca().tick_params(axis="both", labelsize=12)
plt.gca().set_xticks(list(range(0, 2_000 + 1, 200)))
# plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: x / 1_000))
plt.gca().set_xlabel("Learnable parameter count", fontsize=12)
plt.gca().set_ylabel("Accuracy", fontsize=12)

plt.tight_layout()

plt.savefig("mnist_optuna_viz.pdf", bbox_inches="tight")