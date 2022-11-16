# %% just compute a couple of sizes


def made_size(l, h, d, k=None):
    return int(1.5 * d * h + 0.5 * (l - 1) * h * h)


def nvp_size(l, h, d, k=10):
    return int(2 * k * d * h + 2 * k * (l - 1) * h * h)


def maf_size(l, h, d, k=10):
    return int(1.5 * k * d * h + 0.5 * k * (l - 1) * h * h)


def format_as_str(num):
    if num / 1e9 > 1:
        factor, suffix = 1e9, "B"
    elif num / 1e6 > 1:
        factor, suffix = 1e6, "M"
    elif num / 1e3 > 1:
        factor, suffix = 1e3, "K"
    else:
        factor, suffix = 1e0, ""

    num_factored = num / factor

    if num_factored / 1e2 > 1 or True:
        num_rounded = str(int(round(num_factored)))
    elif num_factored / 1e1 > 1:
        num_rounded = f"{num_factored:.1f}"
    else:
        num_rounded = f"{num_factored:.2f}"

    return f"{num_rounded}{suffix} % {num}"


datasets = {
    "power": {"l": 2, "h": 100, "d": 6},
    "gas": {"l": 2, "h": 100, "d": 8},
    "hepmass": {"l": 2, "h": 512, "d": 21},
    "miniboone": {"l": 2, "h": 512, "d": 43},
    "bsds300": {"l": 2, "h": 1024, "d": 63},
    "mnist": {"l": 1, "h": 1024, "d": 784, "k": 10},
    "cifar": {"l": 2, "h": 2048, "d": 3072, "k": 10},
}

networks = {
    "MADE": made_size,
    "RealNVP": nvp_size,
    "MAF": maf_size,
}

for net, handle in networks.items():
    print(net)
    for dset, s_kwargs in datasets.items():
        print(f"{dset}: #params: {format_as_str(handle(**s_kwargs))}")
    print("\n")


# miniboone ffjord: 820613