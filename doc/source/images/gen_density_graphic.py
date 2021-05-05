from matplotlib import pyplot as plt

plt.rcParams["text.usetex"] = True

r_cut = 10
diameter = 3

x = [0, r_cut - diameter, r_cut, r_cut + diameter, 2 * r_cut]
y = [1, 1, 0.5, 0, 0]

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(x, y, color="k")
ax.set_xticks(x[1:4])
ax.set_yticks(y[1:4])
fontdict = {"fontsize": 14}
ax.set_xticklabels(
    [r"$r_{cut} - \frac{d}{2}$", r"$r_{cut}$", r"$r_{cut} + \frac{d}{2}$"], fontdict
)
ax.set_yticklabels(["$1.0$", "$0.5$", "$0$"], fontdict)
ax.hlines(0.5, 0, r_cut, linestyles="dashed")
ax.vlines(r_cut, 0, 0.5, linestyles="dashed")
ax.set_xlim([0, 2 * r_cut])
ax.set_ylim([0, 1.05])
ax.set_title("Fractional neighbor counting", fontsize=20)

fig.savefig("density.png")
