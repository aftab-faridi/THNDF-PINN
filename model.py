import os
import math
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pyDOE import lhs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import griddata
from scipy import stats

# Reproducibility
torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working on {device}")

# -------------------------
# Parameter labels (LaTeX-ready)
# -------------------------
param_labels = {
    "Lambda": r"\Lambda",
    "M": r"M",
    "Nm": r"N_m",
    "Bc": r"B_c",
    "Pr": r"Pr",
    "Rd": r"R_d",
    "Ec": r"Ec",
    "Sc": r"Sc",
    "delta": r"\delta",
    "Td": r"T_d",
    "E": r"E",
    "Sv": r"S_v",
    "St": r"S_t",
    "lp": r"\lambda_p",
    "phi_cu": r"\phi_{Cu}",
    "phi_sic": r"\phi_{SiC}",
    "phi_tio2": r"\phi_{TiO_2}",
}


# -------------------------
# Utilities: compute B params
# -------------------------
def compute_B_params(phi_cu, phi_sic, phi_tio2):
    rho_f, cp_f, k_f, _mu_f, sigma_f = 884, 1910, 0.144, 0.081, 2e-4
    rhocp_f = rho_f * cp_f

    rho_cu, cp_cu, k_cu, sigma_cu = 8933, 385, 401, 5.96e7
    rho_sic, cp_sic, k_sic, sigma_sic = 3370, 1340, 150, 140
    rho_ti, cp_ti, k_ti, sigma_ti = 4230, 692, 8.4, 1e-12

    B1 = (1 - phi_cu) ** 2.5 * (1 - phi_sic) ** 2.5 * (1 - phi_tio2) ** 2.5
    B2 = (
        (1 - phi_cu) * (1 - phi_sic) * (1 - phi_tio2)
        + (1 - phi_sic) * (1 - phi_tio2) * phi_cu * rho_cu / rho_f
        + (1 - phi_tio2) * phi_sic * rho_sic / rho_f
        + phi_tio2 * rho_ti / rho_f
    )
    B3 = (
        (1 - phi_cu) * (1 - phi_sic) * (1 - phi_tio2)
        + (1 - phi_sic) * (1 - phi_tio2) * phi_cu * (rho_cu * cp_cu) / rhocp_f
        + (1 - phi_tio2) * phi_sic * (rho_sic * cp_sic) / rhocp_f
        + phi_tio2 * (rho_ti * cp_ti) / rhocp_f
    )

    def maxwell_garnett(k_base, k_particle, phi):
        return (
            k_base
            * (k_particle + 2 * k_base - 2 * phi * (k_base - k_particle))
            / (k_particle + 2 * k_base + phi * (k_base - k_particle))
        )

    k_thnf = maxwell_garnett(k_f, k_cu, phi_cu)
    k_thnf = maxwell_garnett(k_thnf, k_sic, phi_sic)
    k_thnf = maxwell_garnett(k_thnf, k_ti, phi_tio2)
    B4 = k_thnf / k_f

    sigma_thnf = sigma_f
    for phi, sigma in [(phi_cu, sigma_cu), (phi_sic, sigma_sic), (phi_tio2, sigma_ti)]:
        sigma_thnf = (
            sigma_thnf
            * (sigma + 2 * sigma_thnf - 2 * phi * (sigma_thnf - sigma))
            / (sigma + 2 * sigma_thnf + phi * (sigma_thnf - sigma))
        )
    B5 = sigma_thnf / sigma_f
    B6 = 1.0 / ((1 - phi_cu) * (1 - phi_sic) * (1 - phi_tio2))

    return B1, B2, B3, B4, B5, B6


# -------------------------
# PINN model
# -------------------------
class PINN_Param(nn.Module):
    def __init__(self, layers, param_dim):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        )
        self.param_dim = param_dim

    def forward(self, eta_and_params):
        a = eta_and_params
        for i in range(len(self.linears) - 1):
            a = self.activation(self.linears[i](a))
        return self.linears[-1](a)

    def network_prediction(self, eta, params):
        inp = torch.cat([eta, params], dim=1)
        out = self.forward(inp)
        f, fp, g, theta, phi = (
            out[:, 0:1],
            out[:, 1:2],
            out[:, 2:3],
            out[:, 3:4],
            out[:, 4:5],
        )
        return f, fp, g, theta, phi

    def get_derivative(self, y, x, n=1):
        if n == 0:
            return y
        dy_dx = torch.autograd.grad(
            y,
            x,
            torch.ones_like(y).to(device),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        return self.get_derivative(dy_dx, x, n - 1)


# -------------------------
# Physics residuals + losses
# -------------------------
def physics_residuals(model, eta, params_row, params_dict):
    f, fp, g, theta, phi = model.network_prediction(
        eta, params_row.repeat(eta.shape[0], 1)
    )
    fpp = model.get_derivative(fp, eta, 1)
    fppp = model.get_derivative(fpp, eta, 1)
    gp = model.get_derivative(g, eta, 1)
    gpp = model.get_derivative(gp, eta, 1)
    thetap = model.get_derivative(theta, eta, 1)
    thetapp = model.get_derivative(thetap, eta, 1)
    phip = model.get_derivative(phi, eta, 1)
    phipp = model.get_derivative(phip, eta, 1)

    B1, B2, B3, B4, B5, B6 = (
        params_dict["B1"],
        params_dict["B2"],
        params_dict["B3"],
        params_dict["B4"],
        params_dict["B5"],
        params_dict["B6"],
    )
    Lambda, lp, M, Nm, Bc, Pr, Rd, Ec, Sc, delta, E, Td = (
        params_dict["Lambda"],
        params_dict["lp"],
        params_dict["M"],
        params_dict["Nm"],
        params_dict["Bc"],
        params_dict["Pr"],
        params_dict["Rd"],
        params_dict["Ec"],
        params_dict["Sc"],
        params_dict["delta"],
        params_dict["E"],
        params_dict["Td"],
    )
    n_exp = params_dict.get("n", 1.0)

    one_over_1plusinvLambda = 1.0 / (1.0 + 1.0 / Lambda)
    pref_B1B2 = (B1 * B2) / (2.0 * (1.0 + 1.0 / Lambda))
    pref_M = (B1 * B5) * one_over_1plusinvLambda

    RHS_f = (
        (lp / 2.0) * fp
        + pref_M * (M * fp)
        + pref_B1B2 * (fp**2 - 2.0 * f * fpp - g**2 - Nm * (theta + Bc * phi))
    )
    RHS_g = (
        (lp / 2.0) * g
        + pref_M * (M * g)
        + pref_B1B2 * (2.0 * fp * g - 2.0 * f * gp - Nm * (theta + Bc * phi))
    )
    RHS_theta = -(B3 * Pr / (B4 + (4.0 / 3.0) * Rd)) * f * thetap - (
        (1.0 + 1.0 / Lambda) * Pr * Ec / (B1 * (B4 + (4.0 / 3.0) * Rd))
    ) * (fpp**2)
    RHS_phi = (
        B6
        * Sc
        * (
            delta
            * phi
            * (1.0 + Td * theta) ** n_exp
            * torch.exp(-E / (1.0 + Td * theta))
            - f * phip
        )
    )

    # Return residuals (shape: N_domain x 1)
    return fppp - RHS_f, gpp - RHS_g, thetapp - RHS_theta, phipp - RHS_phi


def ic_slip_loss(model, eta_ic, params_row, ic_values, params_dict):
    f, fp, g, theta, phi = model.network_prediction(
        eta_ic, params_row.repeat(eta_ic.shape[0], 1)
    )
    fpp = model.get_derivative(fp, eta_ic, 1)
    gp = model.get_derivative(g, eta_ic, 1)
    thetap = model.get_derivative(theta, eta_ic, 1)
    Sv, St = params_dict["Sv"], params_dict["St"]
    return (
        torch.mean((f - ic_values["f_ic"]) ** 2)
        + torch.mean((fp - Sv * fpp) ** 2)
        + torch.mean((g - (1.0 + Sv * gp)) ** 2)
        + torch.mean((theta - (1.0 + St * thetap)) ** 2)
        + torch.mean((phi - ic_values["phi_ic"]) ** 2)
    )


def bc_farfield_loss(model, eta_bc, params_row, bc_values):
    f, fp, g, theta, phi = model.network_prediction(
        eta_bc, params_row.repeat(eta_bc.shape[0], 1)
    )
    return (
        torch.mean((bc_values["f_bc"] - f) ** 2)
        + torch.mean((bc_values["fp_bc"] - fp) ** 2)
        + torch.mean((bc_values["g_bc"] - g) ** 2)
        + torch.mean((bc_values["theta_bc"] - theta) ** 2)
        + torch.mean((bc_values["phi_bc"] - phi) ** 2)
    )


def compute_engineering_quantities(model, params_row, params_dict):
    # Need gradients; ensure input tensor requires grad
    eta_zero = torch.zeros(1, 1, requires_grad=True).to(device)
    f0, fp0, g0, theta0, phi0 = model.network_prediction(eta_zero, params_row)
    fpp0 = model.get_derivative(fp0, eta_zero, 1)
    gp0 = model.get_derivative(g0, eta_zero, 1)
    thetap0 = model.get_derivative(theta0, eta_zero, 1)
    phip0 = model.get_derivative(phi0, eta_zero, 1)

    B1, B4, Rd, B6 = (
        params_dict["B1"],
        params_dict["B4"],
        params_dict["Rd"],
        params_dict["B6"],
    )
    Cf_Re_half = (
        (1.0 / B1) * (1.0 + 1.0 / params_dict["Lambda"]) * torch.sqrt(fpp0**2 + gp0**2)
    )
    Nu_Re_neg_half = -B4 * (1.0 + 4.0 / 3.0 * Rd) * thetap0
    Sh_Re_neg_half = -B6 * phip0
    return Cf_Re_half.squeeze(), Nu_Re_neg_half.squeeze(), Sh_Re_neg_half.squeeze()


# -------------------------
# Training with train/test split and extended history
# -------------------------
def train_pinn_on_csv(
    csv_path,
    layers=[1 + 16, 64, 64, 64, 5],
    eta_min=0.0,
    eta_max=6.0,
    N_ic=150,
    N_bc=150,
    N_domain=2000,
    epochs=1500,
    lr=1e-3,
    test_size=0.5,
    max_rows=100,
):
    data = pd.read_csv(csv_path)

    # restrict to max_rows (random sample)
    if len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=42).reset_index(drop=True)
    print("CSV rows (restricted):", len(data))

    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    print(f"Train rows: {len(train_data)}, Test rows: {len(test_data)}")

    target_cols = ["Cf", "Nu", "Sh"]
    param_cols = [c for c in data.columns if c not in target_cols]

    # Build model (input is eta (1) + params)
    mlp_layers = [1 + len(param_cols)] + layers[1:]
    model = PINN_Param(mlp_layers, len(param_cols)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # IC and BC collocation points (shared)
    eta_ic = torch.zeros(N_ic, 1, requires_grad=True).to(device)
    ic_values = {
        "f_ic": torch.zeros(N_ic, 1).to(device),
        "fp_ic": torch.zeros(N_ic, 1).to(device),
        "g_ic": torch.ones(N_ic, 1).to(device),
        "theta_ic": torch.ones(N_ic, 1).to(device),
        "phi_ic": torch.ones(N_ic, 1).to(device),
    }
    eta_bc = torch.full((N_bc, 1), eta_max, requires_grad=True).to(device)
    bc_values = {
        "f_bc": torch.zeros(N_bc, 1).to(device),
        "fp_bc": torch.zeros(N_bc, 1).to(device),
        "g_bc": torch.zeros(N_bc, 1).to(device),
        "theta_bc": torch.zeros(N_bc, 1).to(device),
        "phi_bc": torch.zeros(N_bc, 1).to(device),
    }

    # history dicts
    history = {
        "epoch": [],
        "train_total": [],
        "train_data": [],
        "train_bc": [],
        "train_ic": [],
        "train_pde": [],
        "test_total": [],
        "test_data": [],
        "test_bc": [],
        "test_ic": [],
        "test_pde": [],
    }

    # convert test_data & train_data to list of rows for faster iteration
    train_rows = list(train_data.itertuples(index=False, name=None))
    test_rows = list(test_data.itertuples(index=False, name=None))
    col_names = list(train_data.columns)

    # helper to convert tuple row to dict (faster than pandas inside loops)
    def rowtuple_to_dict(tup):
        return {col_names[i]: tup[i] for i in range(len(col_names))}

    # training loop with epoch-wise evaluation on test set
    for epoch in range(epochs):
        model.train()
        # accumulate train-component sums
        sum_total = 0.0
        sum_data = 0.0
        sum_bc = 0.0
        sum_ic = 0.0
        sum_pde = 0.0

        for tup in train_rows:
            row = rowtuple_to_dict(tup)
            phi_cu, phi_sic, phi_tio2 = row["phi_cu"], row["phi_sic"], row["phi_tio2"]
            B1, B2, B3, B4, B5, B6 = compute_B_params(phi_cu, phi_sic, phi_tio2)
            params_dict = {
                k: float(row[k]) for k in row.keys() if k not in ["Cf", "Nu", "Sh"]
            }
            params_dict.update(
                {
                    "B1": float(B1),
                    "B2": float(B2),
                    "B3": float(B3),
                    "B4": float(B4),
                    "B5": float(B5),
                    "B6": float(B6),
                    "n": 1.0,
                }
            )
            param_vector = (
                torch.tensor([row[c] for c in param_cols], dtype=torch.float32)
                .view(1, -1)
                .to(device)
            )

            # domain collocation points
            eta_domain = (
                torch.tensor(
                    eta_min + (eta_max - eta_min) * lhs(1, N_domain), requires_grad=True
                )
                .float()
                .to(device)
            )

            optimizer.zero_grad()
            pde1, pde2, pde3, pde4 = physics_residuals(
                model, eta_domain, param_vector, params_dict
            )
            mse_pde = (
                torch.mean(pde1**2)
                + torch.mean(pde2**2)
                + torch.mean(pde3**2)
                + torch.mean(pde4**2)
            )
            mse_ic = ic_slip_loss(model, eta_ic, param_vector, ic_values, params_dict)
            mse_bc = bc_farfield_loss(model, eta_bc, param_vector, bc_values)
            Cf_pred, Nu_pred, Sh_pred = compute_engineering_quantities(
                model, param_vector, params_dict
            )
            # row["Cf"] etc are numbers (float)
            mse_data = (
                (Cf_pred - float(row["Cf"])) ** 2
                + (Nu_pred - float(row["Nu"])) ** 2
                + (Sh_pred - float(row["Sh"])) ** 2
            )

            loss = mse_ic + mse_bc + mse_pde + mse_data
            loss.backward()
            optimizer.step()

            # accumulate floats (detach)
            sum_total += loss.item()
            sum_data += mse_data.item()
            sum_bc += mse_bc.item()
            sum_ic += mse_ic.item()
            sum_pde += mse_pde.item()

        # compute averages for train
        n_train = len(train_rows)
        avg_train_total = sum_total / n_train
        avg_train_data = sum_data / n_train
        avg_train_bc = sum_bc / n_train
        avg_train_ic = sum_ic / n_train
        avg_train_pde = sum_pde / n_train

        # Evaluate on test set (epoch-wise) — keep gradients enabled because compute_engineering_quantities needs them
        model.eval()
        # Torch: allow gradients (we will not call backward here)
        sum_total_t = 0.0
        sum_data_t = 0.0
        sum_bc_t = 0.0
        sum_ic_t = 0.0
        sum_pde_t = 0.0

        # Keep autograd enabled for derivatives in compute_engineering_quantities
        with torch.set_grad_enabled(True):
            for tup in test_rows:
                row = rowtuple_to_dict(tup)
                phi_cu, phi_sic, phi_tio2 = (
                    row["phi_cu"],
                    row["phi_sic"],
                    row["phi_tio2"],
                )
                B1, B2, B3, B4, B5, B6 = compute_B_params(phi_cu, phi_sic, phi_tio2)
                params_dict = {
                    k: float(row[k]) for k in row.keys() if k not in ["Cf", "Nu", "Sh"]
                }
                params_dict.update(
                    {
                        "B1": float(B1),
                        "B2": float(B2),
                        "B3": float(B3),
                        "B4": float(B4),
                        "B5": float(B5),
                        "B6": float(B6),
                        "n": 1.0,
                    }
                )
                param_vector = (
                    torch.tensor([row[c] for c in param_cols], dtype=torch.float32)
                    .view(1, -1)
                    .to(device)
                )
                eta_domain = (
                    torch.tensor(
                        eta_min + (eta_max - eta_min) * lhs(1, N_domain),
                        requires_grad=True,
                    )
                    .float()
                    .to(device)
                )

                pde1, pde2, pde3, pde4 = physics_residuals(
                    model, eta_domain, param_vector, params_dict
                )
                mse_pde = (
                    torch.mean(pde1**2)
                    + torch.mean(pde2**2)
                    + torch.mean(pde3**2)
                    + torch.mean(pde4**2)
                )
                mse_ic = ic_slip_loss(
                    model, eta_ic, param_vector, ic_values, params_dict
                )
                mse_bc = bc_farfield_loss(model, eta_bc, param_vector, bc_values)
                Cf_pred, Nu_pred, Sh_pred = compute_engineering_quantities(
                    model, param_vector, params_dict
                )
                mse_data = (
                    (Cf_pred - float(row["Cf"])) ** 2
                    + (Nu_pred - float(row["Nu"])) ** 2
                    + (Sh_pred - float(row["Sh"])) ** 2
                )

                sum_total_t += (mse_ic + mse_bc + mse_pde + mse_data).item()
                sum_data_t += mse_data.item()
                sum_bc_t += mse_bc.item()
                sum_ic_t += mse_ic.item()
                sum_pde_t += mse_pde.item()

        n_test = len(test_rows)
        avg_test_total = sum_total_t / n_test
        avg_test_data = sum_data_t / n_test
        avg_test_bc = sum_bc_t / n_test
        avg_test_ic = sum_ic_t / n_test
        avg_test_pde = sum_pde_t / n_test

        # store history
        history["epoch"].append(epoch)
        history["train_total"].append(avg_train_total)
        history["train_data"].append(avg_train_data)
        history["train_bc"].append(avg_train_bc)
        history["train_ic"].append(avg_train_ic)
        history["train_pde"].append(avg_train_pde)

        history["test_total"].append(avg_test_total)
        history["test_data"].append(avg_test_data)
        history["test_bc"].append(avg_test_bc)
        history["test_ic"].append(avg_test_ic)
        history["test_pde"].append(avg_test_pde)

        if epoch % max(1, epochs // 10) == 0:
            print(
                f"Epoch {epoch}/{epochs} | Train total: {avg_train_total:.3e} | Test total: {avg_test_total:.3e}"
            )

    # return model, column names, train/test data frames, history
    return model, param_cols, train_data, test_data, history


# -------------------------
# Visualization & evaluation helpers (Seaborn + Matplotlib)
# -------------------------
# Set up matplotlib and seaborn styles
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["legend.fontsize"] = 12
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["grid.linewidth"] = 1
plt.rcParams["lines.linewidth"] = 2
sns.set_style("white")
sns.set_palette("husl")


def apply_bold_formatting(ax=None):
    """Apply bold formatting to current plot or specified axes."""
    if ax is None:
        ax = plt.gca()

    # Make all text elements bold
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.tick_params(axis="both", which="minor", labelsize=10)

    # Apply bold formatting to tick labels
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Bold grid lines
    ax.grid(True, alpha=0.3, linewidth=1.2)

    # Bold spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    return ax


def _pad_range(arr, pad_frac=0.04):
    """Return [min-pad, max+pad] with pad_frac fraction of span (safe for constant arrays)."""
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if np.isclose(mn, mx):
        pad = abs(mn) * 0.05 if not np.isclose(mn, 0.0) else 1.0
    else:
        pad = (mx - mn) * pad_frac
    return [mn - pad, mx + pad]


def save_dual_line_log(x, ys, labels, title, xlabel, ylabel, filename_base):
    """Create log-scale line plots using BOTH seaborn and matplotlib."""
    # Seaborn version
    plt.figure(figsize=(10, 6))
    for y, lab in zip(ys, labels):
        sns.lineplot(x=x, y=y, label=lab)
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    # plt.title(title, fontweight="bold", fontsize=14)
    plt.yscale("log")
    plt.legend(prop={"weight": "bold"})
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    base, ext = os.path.splitext(filename_base)
    # Add title to filename
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    if ext == "":
        seaborn_filename = base + f"_{safe_title}_log_scale_seaborn.png"
    else:
        seaborn_filename = base + f"_{safe_title}_log_scale_seaborn" + ext
    plt.tight_layout()
    plt.savefig(seaborn_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_filename}")

    # Matplotlib version
    plt.figure(figsize=(8, 5))
    for y, lab in zip(ys, labels):
        plt.plot(x, y, label=lab)
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    # plt.title(title, fontweight="bold", fontsize=14)
    plt.yscale("log")
    plt.legend(prop={"weight": "bold"})
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    if ext == "":
        matplotlib_filename = base + f"_{safe_title}_log_scale_matplotlib.png"
    else:
        matplotlib_filename = base + f"_{safe_title}_log_scale_matplotlib" + ext
    plt.tight_layout()
    plt.savefig(matplotlib_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_filename}")


def save_dual_gradient_analysis(x, ys, labels, title, xlabel, filename_base):
    """Create gradient analysis plots showing rate of change in both linear and log scale."""
    # Calculate gradients (rate of change)
    gradients = []
    for y in ys:
        grad = np.gradient(y, x)
        gradients.append(grad)

    # Linear scale version
    plt.figure(figsize=(10, 6))
    for grad, lab in zip(gradients, labels):
        sns.lineplot(x=x, y=grad, label=f"{lab} Gradient")
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel("Gradient", fontweight="bold")
    # plt.title(title, fontweight="bold", fontsize=14)
    plt.legend(prop={"weight": "bold"})
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    base, ext = os.path.splitext(filename_base)
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    seaborn_linear = base + f"_{safe_title}_gradient_analysis_linear_seaborn.png"
    plt.tight_layout()
    plt.savefig(seaborn_linear, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_linear}")

    # Matplotlib linear version
    plt.figure(figsize=(8, 5))
    for grad, lab in zip(gradients, labels):
        plt.plot(x, grad, label=f"{lab} Gradient")
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel("Gradient", fontweight="bold")
    # plt.title(title, fontweight="bold", fontsize=14)
    plt.legend(prop={"weight": "bold"})
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    matplotlib_linear = base + f"_{safe_title}_gradient_analysis_linear_matplotlib.png"
    plt.tight_layout()
    plt.savefig(matplotlib_linear, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_linear}")

    # Log scale version (seaborn)
    plt.figure(figsize=(10, 6))
    for grad, lab in zip(gradients, labels):
        # Use absolute values for log scale
        abs_grad = np.abs(grad)
        abs_grad[abs_grad <= 0] = 1e-10  # Replace zeros/negatives with small positive
        sns.lineplot(x=x, y=abs_grad, label=f"{lab} Absolute Gradient")
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel("Absolute Gradient", fontweight="bold")
    # plt.title(title + " (Log Scale)", fontweight="bold", fontsize=14)
    plt.yscale("log")
    plt.legend(prop={"weight": "bold"})
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    seaborn_log = base + f"_{safe_title}_gradient_analysis_log_scale_seaborn.png"
    plt.tight_layout()
    plt.savefig(seaborn_log, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_log}")

    # Matplotlib log version
    plt.figure(figsize=(8, 5))
    for grad, lab in zip(gradients, labels):
        abs_grad = np.abs(grad)
        abs_grad[abs_grad <= 0] = 1e-10
        plt.plot(x, abs_grad, label=f"{lab} Absolute Gradient")
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel("Absolute Gradient", fontweight="bold")
    # plt.title(title + " (Log Scale)", fontweight="bold", fontsize=14)
    plt.yscale("log")
    plt.legend(prop={"weight": "bold"})
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    matplotlib_log = base + f"_{safe_title}_gradient_analysis_log_scale_matplotlib.png"
    plt.tight_layout()
    plt.savefig(matplotlib_log, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_log}")


def save_dual_radar_chart(metrics_df, title, filename_base):
    """Create radar charts for performance metrics comparison."""
    import math

    # Normalize metrics for radar chart (0-1 scale)
    metrics_norm = metrics_df.copy()
    for col in ["MSE", "RMSE", "MAE"]:
        # For error metrics, invert so higher is better (1 - normalized_error)
        max_val = metrics_norm[col].max()
        if max_val > 0:
            metrics_norm[col] = 1 - (metrics_norm[col] / max_val)
        else:
            metrics_norm[col] = 1.0

    # R2 can be negative, so normalize differently
    r2_min = metrics_norm["R2"].min()
    r2_max = metrics_norm["R2"].max()
    if r2_max > r2_min:
        metrics_norm["R2"] = (metrics_norm["R2"] - r2_min) / (r2_max - r2_min)
    else:
        metrics_norm["R2"] = 0.5

    # Setup radar chart parameters
    categories = ["RMSE", "MAE", "MSE", "R²"]
    N = len(categories)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle

    # Seaborn version
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    colors = sns.color_palette("husl", len(metrics_norm))

    for i, (_, row) in enumerate(metrics_norm.iterrows()):
        values = [row["RMSE"], row["MAE"], row["MSE"], row["R2"]]
        values += values[:1]  # Complete the circle

        ax.plot(
            angles, values, "o-", linewidth=2, label=row["Variable"], color=colors[i]
        )
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.01)  # Add extra space to prevent label overlap
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # Position labels further from the plot
    ax.tick_params(axis="x", pad=20)  # Add padding to x-axis labels

    # Adjust legend position to avoid overlap
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.2, 1.1),
        fontsize=10,
        prop={"weight": "bold"},
    )

    base, ext = os.path.splitext(filename_base)
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    seaborn_filename = base + f"_{safe_title}_seaborn.png"
    plt.tight_layout()
    plt.savefig(seaborn_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_filename}")

    # Matplotlib version
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection="polar"))
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics_norm)))

    for i, (_, row) in enumerate(metrics_norm.iterrows()):
        values = [row["RMSE"], row["MAE"], row["MSE"], row["R2"]]
        values += values[:1]

        ax.plot(
            angles, values, "o-", linewidth=2, label=row["Variable"], color=colors[i]
        )
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.2)  # Add extra space to prevent label overlap
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)

    # Position labels further from the plot
    ax.tick_params(axis="x", pad=20)  # Add padding to x-axis labels

    # Adjust legend position to avoid overlap
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.2, 1.1),
        fontsize=10,
        prop={"weight": "bold"},
    )

    matplotlib_filename = base + f"_{safe_title}_matplotlib.png"
    plt.tight_layout()
    plt.savefig(matplotlib_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_filename}")


def save_dual_qq_plots(y_true, y_pred, var_name, filename_base):
    """Create Q-Q plots for residual analysis."""

    residuals = y_pred - y_true

    # Seaborn version
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_xlabel("Theoretical Quantiles", fontweight="bold")
    ax1.set_ylabel("Sample Quantiles", fontweight="bold")
    # ax1.set_title(f"Q-Q Plot: {var_name} Residuals", fontweight="bold", fontsize=14)
    ax1.tick_params(axis="both", labelsize=10)
    # Apply bold formatting to tick labels
    for label in ax1.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax1.get_yticklabels():
        label.set_fontweight("bold")

    # Histogram of residuals
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_xlabel("Residuals", fontweight="bold")
    ax2.set_ylabel("Frequency", fontweight="bold")
    # ax2.set_title(f"Residuals Distribution: {var_name}", fontweight="bold", fontsize=14)
    ax2.tick_params(axis="both", labelsize=10)
    # Apply bold formatting to tick labels
    for label in ax2.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")

    base, ext = os.path.splitext(filename_base)
    safe_var_name = (
        var_name.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    )
    seaborn_filename = base + f"_qq_plot_{safe_var_name}_residuals_seaborn.png"
    plt.tight_layout()
    plt.savefig(seaborn_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_filename}")

    # Matplotlib version
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax1)
    ax1.set_xlabel("Theoretical Quantiles", fontweight="bold")
    ax1.set_ylabel("Sample Quantiles", fontweight="bold")
    # ax1.set_title(f"Q-Q Plot: {var_name} Residuals", fontweight="bold", fontsize=14)
    ax1.tick_params(axis="both", labelsize=10)
    # Apply bold formatting to tick labels
    for label in ax1.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax1.get_yticklabels():
        label.set_fontweight("bold")

    # Histogram of residuals
    ax2.hist(residuals, bins=20, alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Residuals", fontweight="bold")
    ax2.set_ylabel("Frequency", fontweight="bold")
    # ax2.set_title(f"Residuals Distribution: {var_name}", fontweight="bold", fontsize=14)
    ax2.tick_params(axis="both", labelsize=10)
    # Apply bold formatting to tick labels
    for label in ax2.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax2.get_yticklabels():
        label.set_fontweight("bold")

    matplotlib_filename = base + f"_qq_plot_{safe_var_name}_residuals_matplotlib.png"
    plt.tight_layout()
    plt.savefig(matplotlib_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_filename}")


def save_dual_feature_importance(
    param_cols, train_df, test_df, test_preds_df, filename_base
):
    """Create feature importance analysis using correlation with predictions."""
    # Calculate correlations between parameters and prediction errors
    importance_data = []

    for var in ["Cf", "Nu", "Sh"]:
        pred_col = f"{var}_pred"
        true_col = f"{var}_true"

        if pred_col in test_preds_df.columns and true_col in test_preds_df.columns:
            # Calculate absolute prediction errors
            errors = np.abs(test_preds_df[pred_col] - test_preds_df[true_col])

            # For each parameter, calculate correlation with errors
            for param in param_cols:
                if param in test_df.columns:
                    # Get parameter values for test set (use test_df directly)
                    param_values = test_df[param].values
                    if len(param_values) == len(errors):
                        corr = np.corrcoef(param_values, errors)[0, 1]
                        importance_data.append(
                            {
                                "Parameter": param,
                                "Variable": var,
                                "Correlation": abs(corr) if not np.isnan(corr) else 0,
                            }
                        )

    if not importance_data:
        print("Warning: Could not calculate feature importance - insufficient data")
        return

    importance_df = pd.DataFrame(importance_data)

    # Aggregate importance across variables
    param_importance = (
        importance_df.groupby("Parameter")["Correlation"]
        .mean()
        .sort_values(ascending=True)
    )

    # Convert parameter names to LaTeX labels
    param_labels_display = []
    param_values_display = []
    for param in param_importance.index:
        if param in param_labels:
            param_labels_display.append(f"${param_labels[param]}$")
        else:
            param_labels_display.append(param)
        param_values_display.append(param_importance[param])

    # Seaborn version
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x=param_values_display,
        y=param_labels_display,
        hue=param_labels_display,
        palette="viridis",
        legend=False,
    )
    plt.xlabel("Average Absolute Correlation with Prediction Errors", fontweight="bold")
    plt.ylabel("Parameters", fontweight="bold")
    # plt.title("Feature Importance Analysis", fontweight="bold", fontsize=14)
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")

    base, ext = os.path.splitext(filename_base)
    seaborn_filename = base + "_feature_importance_analysis_seaborn.png"
    plt.tight_layout()
    plt.savefig(seaborn_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_filename}")

    # Matplotlib version
    plt.figure(figsize=(10, 8))
    plt.barh(
        param_labels_display,
        param_values_display,
        color="skyblue",
        edgecolor="black",
    )
    plt.xlabel("Average Absolute Correlation with Prediction Errors", fontweight="bold")
    plt.ylabel("Parameters", fontweight="bold")
    # plt.title("Feature Importance Analysis", fontweight="bold", fontsize=14)
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")

    matplotlib_filename = base + "_feature_importance_analysis_matplotlib.png"
    plt.tight_layout()
    plt.savefig(matplotlib_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_filename}")


def save_dual_line(x, ys, labels, title, xlabel, ylabel, filename_base):
    """Create line plots using BOTH seaborn and matplotlib."""
    # Seaborn version
    plt.figure(figsize=(10, 6))
    for y, lab in zip(ys, labels):
        sns.lineplot(x=x, y=y, label=lab)
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    # plt.title(title, fontweight="bold", fontsize=14)
    plt.legend(prop={"weight": "bold"})
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    base, ext = os.path.splitext(filename_base)
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    if ext == "":
        seaborn_filename = base + f"_{safe_title}_seaborn.png"
    else:
        seaborn_filename = base + f"_{safe_title}_seaborn" + ext
    plt.tight_layout()
    plt.savefig(seaborn_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_filename}")

    # Matplotlib version
    plt.figure(figsize=(8, 5))
    for y, lab in zip(ys, labels):
        plt.plot(x, y, label=lab)
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    # plt.title(title, fontweight="bold", fontsize=14)
    plt.legend(prop={"weight": "bold"})
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    if ext == "":
        matplotlib_filename = base + f"_{safe_title}_matplotlib.png"
    else:
        matplotlib_filename = base + f"_{safe_title}_matplotlib" + ext
    plt.tight_layout()
    plt.savefig(matplotlib_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_filename}")


def save_dual_scatter(x, y, title, xlabel, ylabel, filename_base, with_line=False):
    """Create scatter plots using BOTH seaborn and matplotlib."""
    # Seaborn version
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y)
    if with_line:
        mn = min(min(x), min(y))
        mx = max(max(x), max(y))
        plt.plot([mn, mx], [mn, mx], "r--", alpha=0.7, label="y = x")
        plt.legend(prop={"weight": "bold"})
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    # plt.title(title, fontweight="bold", fontsize=14)
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    base, ext = os.path.splitext(filename_base)
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    if ext == "":
        seaborn_filename = base + f"_{safe_title}_seaborn.png"
    else:
        seaborn_filename = base + f"_{safe_title}_seaborn" + ext
    plt.tight_layout()
    plt.savefig(seaborn_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_filename}")

    # Matplotlib version
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    if with_line:
        mn = min(min(x), min(y))
        mx = max(max(x), max(y))
        plt.plot([mn, mx], [mn, mx], "r--", alpha=0.7, label="y = x")
        plt.legend(prop={"weight": "bold"})
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel(ylabel, fontweight="bold")
    # plt.title(title, fontweight="bold", fontsize=14)
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    if ext == "":
        matplotlib_filename = base + f"_{safe_title}_matplotlib.png"
    else:
        matplotlib_filename = base + f"_{safe_title}_matplotlib" + ext
    plt.tight_layout()
    plt.savefig(matplotlib_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_filename}")


def save_dual_bar(categories, series, series_labels, title, xlabel, filename_base):
    """Create bar plots using BOTH seaborn and matplotlib."""
    # Seaborn version
    data = []
    for i, (vals, lab) in enumerate(zip(series, series_labels)):
        for j, val in enumerate(vals):
            data.append({"Category": categories[j], "Value": val, "Type": lab})

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Category", y="Value", hue="Type")
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel("Value", fontweight="bold")
    # plt.title(title, fontweight="bold", fontsize=14)
    plt.legend(prop={"weight": "bold"})
    plt.xticks(fontweight="bold")
    plt.yticks(fontweight="bold")
    base, ext = os.path.splitext(filename_base)
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    if ext == "":
        seaborn_filename = base + f"_{safe_title}_seaborn.png"
    else:
        seaborn_filename = base + f"_{safe_title}_seaborn" + ext
    plt.tight_layout()
    plt.savefig(seaborn_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_filename}")

    # Matplotlib version
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(series))
    plt.figure(figsize=(8, 5))
    for i, (vals, lab) in enumerate(zip(series, series_labels)):
        plt.bar(x + i * width, vals, width=width, label=lab)
    plt.xticks(x + width * (len(series) - 1) / 2, categories, fontweight="bold")
    plt.xlabel(xlabel, fontweight="bold")
    plt.ylabel("Value", fontweight="bold")
    # plt.title(title, fontweight="bold", fontsize=14)
    plt.legend(prop={"weight": "bold"})
    plt.yticks(fontweight="bold")
    if ext == "":
        matplotlib_filename = base + f"_{safe_title}_matplotlib.png"
    else:
        matplotlib_filename = base + f"_{safe_title}_matplotlib" + ext
    plt.tight_layout()
    plt.savefig(matplotlib_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_filename}")


# Keep the old single functions for compatibility
def save_seaborn_line(x, ys, labels, title, xlabel, ylabel, filename):
    """Create line plots using seaborn and matplotlib."""
    plt.figure(figsize=(10, 6))
    for y, lab in zip(ys, labels):
        sns.lineplot(x=x, y=y, label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    base, ext = os.path.splitext(filename)
    if ext == "":
        # Add title to filename
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = filename + f"_{safe_title}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def save_seaborn_scatter(x, y, title, xlabel, ylabel, filename, with_line=False):
    """Create scatter plots using seaborn and matplotlib."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=x, y=y)
    if with_line:
        mn = min(min(x), min(y))
        mx = max(max(x), max(y))
        plt.plot([mn, mx], [mn, mx], "r--", alpha=0.7, label="y = x")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    base, ext = os.path.splitext(filename)
    if ext == "":
        # Add title to filename
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = filename + f"_{safe_title}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def save_seaborn_bar(categories, series, series_labels, title, xlabel, filename):
    """Create bar plots using seaborn and matplotlib."""
    # Create DataFrame for seaborn
    data = []
    for i, (vals, lab) in enumerate(zip(series, series_labels)):
        for j, val in enumerate(vals):
            data.append({"Category": categories[j], "Value": val, "Type": lab})

    df = pd.DataFrame(data)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Category", y="Value", hue="Type")
    plt.xlabel(xlabel)
    plt.legend()
    base, ext = os.path.splitext(filename)
    if ext == "":
        # Add title to filename
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = filename + f"_{safe_title}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def save_matplotlib_line(x, ys, labels, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 5))
    for y, lab in zip(ys, labels):
        plt.plot(x, y, label=lab)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    base, ext = os.path.splitext(filename)
    if ext == "":
        # Add title to filename
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = filename + f"_{safe_title}_matplotlib.png"
    else:
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = base + f"_{safe_title}_matplotlib" + ext
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved Matplotlib: {filename}")


def save_matplotlib_scatter(x, y, title, xlabel, ylabel, filename, extra=None):
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y)
    if extra and extra.get("y_eq_x", False):
        mn = min(min(x), min(y))
        mx = max(max(x), max(y))
        plt.plot([mn, mx], [mn, mx], linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    base, ext = os.path.splitext(filename)
    if ext == "":
        # Add title to filename
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = filename + f"_{safe_title}_matplotlib.png"
    else:
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = base + f"_{safe_title}_matplotlib" + ext
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved Matplotlib: {filename}")


def save_matplotlib_bar(categories, series, series_labels, title, xlabel, filename):
    x = np.arange(len(categories))
    width = 0.8 / max(1, len(series))
    plt.figure(figsize=(8, 5))
    for i, (vals, lab) in enumerate(zip(series, series_labels)):
        plt.bar(x + i * width, vals, width=width, label=lab)
    plt.xticks(x + width * (len(series) - 1) / 2, categories)
    plt.xlabel(xlabel)
    plt.legend()
    base, ext = os.path.splitext(filename)
    if ext == "":
        # Add title to filename
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = filename + f"_{safe_title}_matplotlib.png"
    else:
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = base + f"_{safe_title}_matplotlib" + ext
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved Matplotlib: {filename}")


def save_seaborn_3d(x, y, z, title, xlabel, ylabel, zlabel, filename, kind="scatter"):
    """Create 3D plots using seaborn style with matplotlib 3D backend."""
    sns.set_style("white")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if kind == "scatter":
        ax.scatter(x, y, z, c=sns.color_palette("husl", 1)[0], alpha=0.7)
    else:
        ax.scatter(x, y, z, c=sns.color_palette("husl", 1)[0], alpha=0.7)
    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_zlabel(zlabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=14)
    apply_bold_formatting(ax)
    base, ext = os.path.splitext(filename)
    if ext == "":
        # Add title to filename
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = filename + f"_{safe_title}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def save_matplotlib_3d(
    x, y, z, title, xlabel, ylabel, zlabel, filename, kind="scatter"
):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if kind == "scatter":
        ax.scatter(x, y, z)
    else:
        ax.scatter(x, y, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    base, ext = os.path.splitext(filename)
    if ext == "":
        # Add title to filename
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = filename + f"_{safe_title}_matplotlib.png"
    else:
        safe_title = (
            title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
        )
        filename = base + f"_{safe_title}_matplotlib" + ext
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved Matplotlib: {filename}")


def save_dual_3d_surface(
    x, y, z, title, xlabel, ylabel, zlabel, filename_base, grid_size=20
):
    """Create 3D surface plots using BOTH seaborn and matplotlib."""
    # Create meshgrid for surface plotting

    # Create regular grid
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    XI, YI = np.meshgrid(xi, yi)

    # Interpolate z values on regular grid
    ZI = griddata((x, y), z, (XI, YI), method="cubic", fill_value=np.nan)

    # Seaborn version (matplotlib backend with seaborn styling)
    sns.set_style("white")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XI, YI, ZI, cmap="viridis", alpha=0.8, edgecolor="none")
    ax.scatter(x, y, z, c="red", s=50, alpha=0.6)  # Original data points
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    base, ext = os.path.splitext(filename_base)
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    if ext == "":
        seaborn_filename = base + f"_{safe_title}_seaborn.png"
    else:
        seaborn_filename = base + f"_{safe_title}_seaborn" + ext
    plt.tight_layout()
    plt.savefig(seaborn_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_filename}")

    # Matplotlib version
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(XI, YI, ZI, cmap="coolwarm", alpha=0.8, edgecolor="none")
    ax.scatter(x, y, z, c="black", s=50, alpha=0.6)  # Original data points
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    if ext == "":
        matplotlib_filename = base + f"_{safe_title}_matplotlib.png"
    else:
        matplotlib_filename = base + f"_{safe_title}_matplotlib" + ext
    plt.tight_layout()
    plt.savefig(matplotlib_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_filename}")


def save_dual_3d(x, y, z, title, xlabel, ylabel, zlabel, filename_base, kind="scatter"):
    """Create 3D plots using BOTH seaborn and matplotlib."""
    # Seaborn version
    sns.set_style("white")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if kind == "scatter":
        ax.scatter(x, y, z, c=sns.color_palette("husl", 1)[0], alpha=0.7)
    else:
        ax.scatter(x, y, z, c=sns.color_palette("husl", 1)[0], alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    base, ext = os.path.splitext(filename_base)
    safe_title = title.replace(" ", "_").replace("/", "_").replace("\\", "_").lower()
    if ext == "":
        seaborn_filename = base + f"_{safe_title}_seaborn.png"
    else:
        seaborn_filename = base + f"_{safe_title}_seaborn" + ext
    plt.tight_layout()
    plt.savefig(seaborn_filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved Seaborn: {seaborn_filename}")

    # Matplotlib version
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if kind == "scatter":
        ax.scatter(x, y, z)
    else:
        ax.scatter(x, y, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if ext == "":
        matplotlib_filename = base + f"_{safe_title}_matplotlib.png"
    else:
        matplotlib_filename = base + f"_{safe_title}_matplotlib" + ext
    plt.tight_layout()
    plt.savefig(matplotlib_filename, dpi=200)
    plt.close()
    print(f"Saved Matplotlib: {matplotlib_filename}")


def metrics_table(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# -------------------------
# Boundary Condition and Initial Condition Verification
# -------------------------
def verify_boundary_and_initial_conditions(
    model, param_cols, test_df, eta_min=0.0, eta_max=6.0
):
    """
    Verify that the neural network predictions satisfy boundary and initial conditions.
    Print detailed tables comparing NN predictions vs. actual BC/IC values.
    """
    print("\n" + "=" * 80)
    print("BOUNDARY CONDITION AND INITIAL CONDITION VERIFICATION")
    print("=" * 80)

    # Take the first few test cases for detailed verification
    n_cases = min(3, len(test_df))

    for case_idx in range(n_cases):
        row = test_df.iloc[case_idx]
        print(f"\n{'=' * 50}")
        print(f"VERIFICATION CASE {case_idx + 1}")
        print(f"{'=' * 50}")

        # Extract parameters for this case
        param_vector = (
            torch.tensor(row[param_cols].values.astype(np.float32))
            .view(1, -1)
            .to(device)
        )
        B1, B2, B3, B4, B5, B6 = compute_B_params(
            row["phi_cu"], row["phi_sic"], row["phi_tio2"]
        )
        params_dict = {
            k: float(row[k]) for k in row.index if k not in ["Cf", "Nu", "Sh"]
        }
        params_dict.update(
            {
                "B1": float(B1),
                "B2": float(B2),
                "B3": float(B3),
                "B4": float(B4),
                "B5": float(B5),
                "B6": float(B6),
                "n": 1.0,
            }
        )

        # Display key parameters for this case
        print("Parameters for this case:")
        key_params = ["M", "Lambda", "lp", "Pr", "Sc", "phi_cu", "phi_sic", "phi_tio2"]
        for param in key_params:
            if param in params_dict:
                print(f"  {param}: {params_dict[param]:.6f}")

        model.eval()
        with torch.set_grad_enabled(True):
            # ============= INITIAL CONDITIONS (at η = 0) =============
            print(f"\n{'Initial Conditions (at η = 0.0)':^60}")
            print("-" * 60)

            eta_ic = torch.zeros(1, 1, requires_grad=True).to(device)
            f_ic, fp_ic, g_ic, theta_ic, phi_ic = model.network_prediction(
                eta_ic, param_vector
            )
            fpp_ic = model.get_derivative(fp_ic, eta_ic, 1)
            gp_ic = model.get_derivative(g_ic, eta_ic, 1)
            thetap_ic = model.get_derivative(theta_ic, eta_ic, 1)

            # Actual IC values (from problem formulation)
            Sv, St = params_dict["Sv"], params_dict["St"]

            # Create IC verification table
            ic_data = [
                ["Variable", "NN Prediction", "Actual IC", "Slip Condition", "Error"],
                [
                    "f(0)",
                    f"{f_ic.item():.6f}",
                    "0.000000",
                    "f(0) = 0",
                    f"{abs(f_ic.item() - 0.0):.6f}",
                ],
                [
                    "f'(0)",
                    f"{fp_ic.item():.6f}",
                    f"{Sv * fpp_ic.item():.6f}",
                    "f'(0) = Sv·f''(0)",
                    f"{abs(fp_ic.item() - Sv * fpp_ic.item()):.6f}",
                ],
                [
                    "g(0)",
                    f"{g_ic.item():.6f}",
                    f"{1.0 + Sv * gp_ic.item():.6f}",
                    "g(0) = 1 + Sv·g'(0)",
                    f"{abs(g_ic.item() - (1.0 + Sv * gp_ic.item())):.6f}",
                ],
                [
                    "θ(0)",
                    f"{theta_ic.item():.6f}",
                    f"{1.0 + St * thetap_ic.item():.6f}",
                    "θ(0) = 1 + St·θ'(0)",
                    f"{abs(theta_ic.item() - (1.0 + St * thetap_ic.item())):.6f}",
                ],
                [
                    "φ(0)",
                    f"{phi_ic.item():.6f}",
                    "1.000000",
                    "φ(0) = 1",
                    f"{abs(phi_ic.item() - 1.0):.6f}",
                ],
            ]

            # Print IC table
            col_widths = [12, 15, 15, 20, 12]
            for i, row_data in enumerate(ic_data):
                if i == 0:  # Header
                    print(
                        "".join(
                            f"{item:^{col_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )
                    print("-" * sum(col_widths))
                else:
                    print(
                        "".join(
                            f"{item:^{col_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )

            # ============= BOUNDARY CONDITIONS (at η = η_max) =============
            print(f"\n{'Boundary Conditions (at η = 6.0)':^60}")
            print("-" * 60)

            eta_bc = torch.full((1, 1), eta_max, requires_grad=True).to(device)
            f_bc, fp_bc, g_bc, theta_bc, phi_bc = model.network_prediction(
                eta_bc, param_vector
            )

            # Actual BC values (far-field conditions)
            bc_data = [
                ["Variable", "NN Prediction", "Actual BC", "Condition", "Error"],
                [
                    "f(∞)",
                    f"{f_bc.item():.6f}",
                    "0.000000",
                    "f(∞) = 0",
                    f"{abs(f_bc.item() - 0.0):.6f}",
                ],
                [
                    "f'(∞)",
                    f"{fp_bc.item():.6f}",
                    "0.000000",
                    "f'(∞) = 0",
                    f"{abs(fp_bc.item() - 0.0):.6f}",
                ],
                [
                    "g(∞)",
                    f"{g_bc.item():.6f}",
                    "0.000000",
                    "g(∞) = 0",
                    f"{abs(g_bc.item() - 0.0):.6f}",
                ],
                [
                    "θ(∞)",
                    f"{theta_bc.item():.6f}",
                    "0.000000",
                    "θ(∞) = 0",
                    f"{abs(theta_bc.item() - 0.0):.6f}",
                ],
                [
                    "φ(∞)",
                    f"{phi_bc.item():.6f}",
                    "0.000000",
                    "φ(∞) = 0",
                    f"{abs(phi_bc.item() - 0.0):.6f}",
                ],
            ]

            # Print BC table
            for i, row_data in enumerate(bc_data):
                if i == 0:  # Header
                    print(
                        "".join(
                            f"{item:^{col_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )
                    print("-" * sum(col_widths))
                else:
                    print(
                        "".join(
                            f"{item:^{col_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )

            # ============= SUMMARY STATISTICS =============
            print(f"\n{'Condition Satisfaction Summary':^60}")
            print("-" * 60)

            # Calculate average errors
            ic_errors = [
                abs(f_ic.item() - 0.0),
                abs(fp_ic.item() - Sv * fpp_ic.item()),
                abs(g_ic.item() - (1.0 + Sv * gp_ic.item())),
                abs(theta_ic.item() - (1.0 + St * thetap_ic.item())),
                abs(phi_ic.item() - 1.0),
            ]

            bc_errors = [
                abs(f_bc.item() - 0.0),
                abs(fp_bc.item() - 0.0),
                abs(g_bc.item() - 0.0),
                abs(theta_bc.item() - 0.0),
                abs(phi_bc.item() - 0.0),
            ]

            avg_ic_error = np.mean(ic_errors)
            avg_bc_error = np.mean(bc_errors)
            max_ic_error = np.max(ic_errors)
            max_bc_error = np.max(bc_errors)

            summary_data = [
                ["Condition Type", "Avg Error", "Max Error", "Status"],
                [
                    "Initial Conditions",
                    f"{avg_ic_error:.2e}",
                    f"{max_ic_error:.2e}",
                    "Good" if max_ic_error < 1e-2 else "Check",
                ],
                [
                    "Boundary Conditions",
                    f"{avg_bc_error:.2e}",
                    f"{max_bc_error:.2e}",
                    "Good" if max_bc_error < 1e-2 else "Check",
                ],
            ]

            summary_widths = [20, 12, 12, 10]
            for i, row_data in enumerate(summary_data):
                if i == 0:  # Header
                    print(
                        "".join(
                            f"{item:^{summary_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )
                    print("-" * sum(summary_widths))
                else:
                    print(
                        "".join(
                            f"{item:^{summary_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


# -------------------------
# Main script
# -------------------------
if __name__ == "__main__":
    csv_path = "data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Missing data.csv with expected columns.")

    # TRAIN
    model, param_cols, train_df, test_df, history = train_pinn_on_csv(
        csv_path, epochs=100, lr=1e-3, test_size=0.3, max_rows=500
    )

    # After training: collect predictions for train and test sets (for Cf, Nu, Sh)
    def predict_df(df):
        preds = {
            "Cf_pred": [],
            "Nu_pred": [],
            "Sh_pred": [],
            "M": [],
            "lp": [],
            "Cf_true": [],
            "Nu_true": [],
            "Sh_true": [],
        }
        model.eval()
        with torch.set_grad_enabled(True):
            for _, row in df.iterrows():
                param_vector = (
                    torch.tensor(row[param_cols].values.astype(np.float32))
                    .view(1, -1)
                    .to(device)
                )
                B1, B2, B3, B4, B5, B6 = compute_B_params(
                    row["phi_cu"], row["phi_sic"], row["phi_tio2"]
                )
                params_dict = {
                    k: float(row[k]) for k in row.index if k not in ["Cf", "Nu", "Sh"]
                }
                params_dict.update(
                    {
                        "B1": float(B1),
                        "B2": float(B2),
                        "B3": float(B3),
                        "B4": float(B4),
                        "B5": float(B5),
                        "B6": float(B6),
                        "n": 1.0,
                    }
                )
                Cf_pred, Nu_pred, Sh_pred = compute_engineering_quantities(
                    model, param_vector, params_dict
                )
                preds["Cf_pred"].append(Cf_pred.item())
                preds["Nu_pred"].append(Nu_pred.item())
                preds["Sh_pred"].append(Sh_pred.item())
                # magnetic = M, porosity = lp (user specified)
                preds["M"].append(float(params_dict.get("M", np.nan)))
                preds["lp"].append(float(params_dict.get("lp", np.nan)))
                preds["Cf_true"].append(float(row["Cf"]))
                preds["Nu_true"].append(float(row["Nu"]))
                preds["Sh_true"].append(float(row["Sh"]))
        return pd.DataFrame(preds)

    train_preds_df = predict_df(train_df)
    test_preds_df = predict_df(test_df)

    # Pretty table: test set predicted vs true
    compare_df = test_preds_df[
        ["Cf_true", "Cf_pred", "Nu_true", "Nu_pred", "Sh_true", "Sh_pred"]
    ].copy()
    compare_csv = "predicted_vs_true_test.csv"
    compare_df.to_csv(compare_csv, index=False)
    print("\nPredicted vs True (test) — saved to", compare_csv)
    print(compare_df.to_string(index=False, float_format="%.6f"))

    # Metrics for each variable (test)
    metrics_Cf = metrics_table(
        compare_df["Cf_true"].values, compare_df["Cf_pred"].values
    )
    metrics_Nu = metrics_table(
        compare_df["Nu_true"].values, compare_df["Nu_pred"].values
    )
    metrics_Sh = metrics_table(
        compare_df["Sh_true"].values, compare_df["Sh_pred"].values
    )

    metrics_df = pd.DataFrame(
        {
            "Variable": ["Cf", "Nu", "Sh"],
            "MSE": [metrics_Cf["MSE"], metrics_Nu["MSE"], metrics_Sh["MSE"]],
            "RMSE": [metrics_Cf["RMSE"], metrics_Nu["RMSE"], metrics_Sh["RMSE"]],
            "MAE": [metrics_Cf["MAE"], metrics_Nu["MAE"], metrics_Sh["MAE"]],
            "R2": [metrics_Cf["R2"], metrics_Nu["R2"], metrics_Sh["R2"]],
        }
    )
    metrics_df.to_csv("metrics_test.csv", index=False)
    print("\nMetrics (test) — saved to metrics_test.csv")
    print(metrics_df.to_string(index=False, float_format="%.6f"))

    # -------------------------
    # COMPREHENSIVE PLOTTING AND ANALYSIS
    # -------------------------
    # 1) Combined Test + Training graph in a single plot (total loss)
    save_dual_line(
        history["epoch"],
        [history["train_total"], history["test_total"]],
        ["Train Total", "Test Total"],
        "Combined Total Loss (Train vs Test)",
        "Epoch",
        "Loss",
        "combined_total_loss",
    )

    # 2) Separate Test and Training graph (each containing separate curve for Data loss, BC, IC and PDE)
    save_dual_line(
        history["epoch"],
        [
            history["train_data"],
            history["train_bc"],
            history["train_ic"],
            history["train_pde"],
        ],
        ["Data Loss", "BC Loss", "IC Loss", "PDE Loss"],
        "Training Loss Components",
        "Epoch",
        "Loss",
        "train_loss_components",
    )

    save_dual_line(
        history["epoch"],
        [
            history["test_data"],
            history["test_bc"],
            history["test_ic"],
            history["test_pde"],
        ],
        ["Data Loss", "BC Loss", "IC Loss", "PDE Loss"],
        "Test Loss Components",
        "Epoch",
        "Loss",
        "test_loss_components",
    )

    # 3) 4 graphs each containing two curves of Test and Train for these Data loss, BC Loss, IC Loss and PDE Loss respectively.
    comps = [
        ("data", "Data Loss"),
        ("bc", "BC Loss"),
        ("ic", "IC Loss"),
        ("pde", "PDE Loss"),
    ]
    for comp_key, comp_label in comps:
        save_dual_line(
            history["epoch"],
            [history[f"train_{comp_key}"], history[f"test_{comp_key}"]],
            [f"Train {comp_label}", f"Test {comp_label}"],
            f"{comp_label} (Train vs Test)",
            "Epoch",
            "Loss",
            f"{comp_key}_train_vs_test",
        )

    # 4) Scatter plots for Cf, Nu, Sh Predicted and Actual.
    for var in ["Cf", "Nu", "Sh"]:
        save_dual_scatter(
            compare_df[f"{var}_true"].values,
            compare_df[f"{var}_pred"].values,
            f"Scatter Predicted vs Actual: {var}",
            f"{var} Actual",
            f"{var} Predicted",
            f"scatter_{var}",
            with_line=True,
        )

    # 5) Bar plots for metrics (RMSE, MAE, MSE, R2)
    save_dual_bar(
        metrics_df["Variable"].tolist(),
        [
            metrics_df["RMSE"].tolist(),
            metrics_df["MAE"].tolist(),
            metrics_df["MSE"].tolist(),
        ],
        ["RMSE", "MAE", "MSE"],
        "Error Metrics (Test)",
        "Variable",
        "metrics_bar",
    )
    # Remove the redundant Plotly code
    # Bar plots are already handled by save_seaborn_bar above

    # 6) 3D response surface plots for each Cf, Nu and Sh: z = variable, x = M, y = lp
    def make_3d_mesh(df_preds, var_true_col, var_pred_col, name):
        x = df_preds["M"].values
        y = df_preds["lp"].values
        z_true = df_preds[var_true_col].values
        z_pred = df_preds[var_pred_col].values

        # Dual 3D scatter plots (both seaborn and matplotlib versions)
        save_dual_3d(
            x,
            y,
            z_true,
            f"3D Response (True) - {name}",
            f"${param_labels.get('M', 'M')}$",
            f"${param_labels.get('lp', r'\\lambda_p')}$",
            f"${name}$",
            f"3d_{name}_true",
        )
        save_dual_3d(
            x,
            y,
            z_pred,
            f"3D Response (Predicted) - {name}",
            f"${param_labels.get('M', 'M')}$",
            f"${param_labels.get('lp', r'\\lambda_p')}$",
            f"${name}$",
            f"3d_{name}_pred",
        )

        # Dual 3D surface plots
        save_dual_3d_surface(
            x,
            y,
            z_true,
            f"3D Surface (True) - {name}",
            f"${param_labels.get('M', 'M')}$",
            f"${param_labels.get('lp', r'\\lambda_p')}$",
            f"${name}$",
            f"3d_surface_{name}_true",
        )
        save_dual_3d_surface(
            x,
            y,
            z_pred,
            f"3D Surface (Predicted) - {name}",
            f"${param_labels.get('M', 'M')}$",
            f"${param_labels.get('lp', r'\\lambda_p')}$",
            f"${name}$",
            f"3d_surface_{name}_pred",
        )

    make_3d_mesh(test_preds_df, "Cf_true", "Cf_pred", "Cf")
    make_3d_mesh(test_preds_df, "Nu_true", "Nu_pred", "Nu")
    make_3d_mesh(test_preds_df, "Sh_true", "Sh_pred", "Sh")

    # 7) Combined Train+Test loss component plot (overlay train and test components in a single figure)
    save_dual_line(
        history["epoch"],
        [
            history["train_data"],
            history["test_data"],
            history["train_bc"],
            history["test_bc"],
            history["train_ic"],
            history["test_ic"],
            history["train_pde"],
            history["test_pde"],
        ],
        [
            "Train Data",
            "Test Data",
            "Train BC",
            "Test BC",
            "Train IC",
            "Test IC",
            "Train PDE",
            "Test PDE",
        ],
        "All Loss Components (Train & Test)",
        "Epoch",
        "Loss",
        "all_components_train_test",
    )

    # 8) Log-scale versions of all loss plots
    print("\n=== Generating Log-Scale Loss Plots ===")

    # Combined total loss (log scale)
    save_dual_line_log(
        history["epoch"],
        [history["train_total"], history["test_total"]],
        ["Train Total", "Test Total"],
        "Combined Total Loss (Train vs Test)",
        "Epoch",
        "Loss",
        "combined_total_loss",
    )

    # Training loss components (log scale)
    save_dual_line_log(
        history["epoch"],
        [
            history["train_data"],
            history["train_bc"],
            history["train_ic"],
            history["train_pde"],
        ],
        ["Data Loss", "BC Loss", "IC Loss", "PDE Loss"],
        "Training Loss Components",
        "Epoch",
        "Loss",
        "train_loss_components",
    )

    # Test loss components (log scale)
    save_dual_line_log(
        history["epoch"],
        [
            history["test_data"],
            history["test_bc"],
            history["test_ic"],
            history["test_pde"],
        ],
        ["Data Loss", "BC Loss", "IC Loss", "PDE Loss"],
        "Test Loss Components",
        "Epoch",
        "Loss",
        "test_loss_components",
    )

    # Individual component comparisons (log scale)
    comps_log = [
        ("data", "Data Loss"),
        ("bc", "BC Loss"),
        ("ic", "IC Loss"),
        ("pde", "PDE Loss"),
    ]
    for comp_key, comp_label in comps_log:
        save_dual_line_log(
            history["epoch"],
            [history[f"train_{comp_key}"], history[f"test_{comp_key}"]],
            [f"Train {comp_label}", f"Test {comp_label}"],
            f"{comp_label} (Train vs Test)",
            "Epoch",
            "Loss",
            f"{comp_key}_train_vs_test",
        )

    # 9) Loss gradient analysis
    print("\n=== Generating Loss Gradient Analysis ===")

    # Total loss gradient analysis
    save_dual_gradient_analysis(
        history["epoch"],
        [history["train_total"], history["test_total"]],
        ["Train Total", "Test Total"],
        "Total Loss",
        "Epoch",
        "total_loss_gradient",
    )

    # Component loss gradient analysis
    save_dual_gradient_analysis(
        history["epoch"],
        [
            history["train_data"],
            history["train_bc"],
            history["train_ic"],
            history["train_pde"],
        ],
        ["Data", "BC", "IC", "PDE"],
        "Training Loss Components",
        "Epoch",
        "train_components_gradient",
    )

    # 10) Performance metrics radar charts
    print("\n=== Generating Performance Radar Charts ===")
    save_dual_radar_chart(
        metrics_df, "Performance Metrics Comparison", "performance_radar"
    )

    # 11) Feature importance analysis
    print("\n=== Generating Feature Importance Analysis ===")
    save_dual_feature_importance(
        param_cols, train_df, test_df, test_preds_df, "feature_importance"
    )

    # 12) Q-Q plots for residual analysis
    print("\n=== Generating Q-Q Plots for Residual Analysis ===")
    for var in ["Cf", "Nu", "Sh"]:
        if f"{var}_true" in compare_df.columns and f"{var}_pred" in compare_df.columns:
            save_dual_qq_plots(
                compare_df[f"{var}_true"].values,
                compare_df[f"{var}_pred"].values,
                var,
                f"qq_plot_{var}",
            )

    print("\nAll figures saved. Summary files:")
    print(" - predicted_vs_true_test.csv")
    print(" - metrics_test.csv")
    print("\n=== COMPREHENSIVE PLOTTING COMPLETE ===")
    print("Generated plot types (all in BOTH Seaborn and Matplotlib versions):")
    print("Loss evolution plots (linear and log scale)")
    print("Loss gradient analysis (linear and log scale)")
    print("Scatter plots with reference lines")
    print("Performance metrics bar charts")
    print("Performance radar charts")
    print("3D scatter plots")
    print("3D surface plots")
    print("Feature importance analysis")
    print("Q-Q plots for residual analysis")
    print(
        f"\nTotal plots generated: {len([f for f in os.listdir('.') if f.endswith('.png')])} files"
    )
    print("Every visualization available in both seaborn and matplotlib styles!")

    # -------------------------
    # BOUNDARY & INITIAL CONDITION VERIFICATION
    # -------------------------
    verify_boundary_and_initial_conditions(model, param_cols, test_df)


# -------------------------
# Boundary Condition and Initial Condition Verification
# -------------------------
def verify_boundary_and_initial_conditions(
    model, param_cols, test_df, eta_min=0.0, eta_max=6.0
):
    """
    Verify that the neural network predictions satisfy boundary and initial conditions.
    Print detailed tables comparing NN predictions vs. actual BC/IC values.
    """
    print("\n" + "=" * 80)
    print("BOUNDARY CONDITION AND INITIAL CONDITION VERIFICATION")
    print("=" * 80)

    # Take the first few test cases for detailed verification
    n_cases = min(3, len(test_df))

    for case_idx in range(n_cases):
        row = test_df.iloc[case_idx]
        print(f"\n{'=' * 50}")
        print(f"VERIFICATION CASE {case_idx + 1}")
        print(f"{'=' * 50}")

        # Extract parameters for this case
        param_vector = (
            torch.tensor(row[param_cols].values.astype(np.float32))
            .view(1, -1)
            .to(device)
        )
        B1, B2, B3, B4, B5, B6 = compute_B_params(
            row["phi_cu"], row["phi_sic"], row["phi_tio2"]
        )
        params_dict = {
            k: float(row[k]) for k in row.index if k not in ["Cf", "Nu", "Sh"]
        }
        params_dict.update(
            {
                "B1": float(B1),
                "B2": float(B2),
                "B3": float(B3),
                "B4": float(B4),
                "B5": float(B5),
                "B6": float(B6),
                "n": 1.0,
            }
        )

        # Display key parameters for this case
        print("Parameters for this case:")
        key_params = ["M", "Lambda", "lp", "Pr", "Sc", "phi_cu", "phi_sic", "phi_tio2"]
        for param in key_params:
            if param in params_dict:
                print(f"  {param}: {params_dict[param]:.6f}")

        model.eval()
        with torch.set_grad_enabled(True):
            # ============= INITIAL CONDITIONS (at η = 0) =============
            print(f"\n{'Initial Conditions (at η = 0.0)':^60}")
            print("-" * 60)

            eta_ic = torch.zeros(1, 1, requires_grad=True).to(device)
            f_ic, fp_ic, g_ic, theta_ic, phi_ic = model.network_prediction(
                eta_ic, param_vector
            )
            fpp_ic = model.get_derivative(fp_ic, eta_ic, 1)
            gp_ic = model.get_derivative(g_ic, eta_ic, 1)
            thetap_ic = model.get_derivative(theta_ic, eta_ic, 1)

            # Actual IC values (from problem formulation)
            Sv, St = params_dict["Sv"], params_dict["St"]

            # Create IC verification table
            ic_data = [
                ["Variable", "NN Prediction", "Actual IC", "Slip Condition", "Error"],
                [
                    "f(0)",
                    f"{f_ic.item():.6f}",
                    "0.000000",
                    "f(0) = 0",
                    f"{abs(f_ic.item() - 0.0):.6f}",
                ],
                [
                    "f'(0)",
                    f"{fp_ic.item():.6f}",
                    f"{Sv * fpp_ic.item():.6f}",
                    "f'(0) = Sv·f''(0)",
                    f"{abs(fp_ic.item() - Sv * fpp_ic.item()):.6f}",
                ],
                [
                    "g(0)",
                    f"{g_ic.item():.6f}",
                    f"{1.0 + Sv * gp_ic.item():.6f}",
                    "g(0) = 1 + Sv·g'(0)",
                    f"{abs(g_ic.item() - (1.0 + Sv * gp_ic.item())):.6f}",
                ],
                [
                    "θ(0)",
                    f"{theta_ic.item():.6f}",
                    f"{1.0 + St * thetap_ic.item():.6f}",
                    "θ(0) = 1 + St·θ'(0)",
                    f"{abs(theta_ic.item() - (1.0 + St * thetap_ic.item())):.6f}",
                ],
                [
                    "φ(0)",
                    f"{phi_ic.item():.6f}",
                    "1.000000",
                    "φ(0) = 1",
                    f"{abs(phi_ic.item() - 1.0):.6f}",
                ],
            ]

            # Print IC table
            col_widths = [12, 15, 15, 20, 12]
            for i, row_data in enumerate(ic_data):
                if i == 0:  # Header
                    print(
                        "".join(
                            f"{item:^{col_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )
                    print("-" * sum(col_widths))
                else:
                    print(
                        "".join(
                            f"{item:^{col_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )

            # ============= BOUNDARY CONDITIONS (at η = η_max) =============
            print(f"\n{'Boundary Conditions (at η = 6.0)':^60}")
            print("-" * 60)

            eta_bc = torch.full((1, 1), eta_max, requires_grad=True).to(device)
            f_bc, fp_bc, g_bc, theta_bc, phi_bc = model.network_prediction(
                eta_bc, param_vector
            )

            # Actual BC values (far-field conditions)
            bc_data = [
                ["Variable", "NN Prediction", "Actual BC", "Condition", "Error"],
                [
                    "f(∞)",
                    f"{f_bc.item():.6f}",
                    "0.000000",
                    "f(∞) = 0",
                    f"{abs(f_bc.item() - 0.0):.6f}",
                ],
                [
                    "f'(∞)",
                    f"{fp_bc.item():.6f}",
                    "0.000000",
                    "f'(∞) = 0",
                    f"{abs(fp_bc.item() - 0.0):.6f}",
                ],
                [
                    "g(∞)",
                    f"{g_bc.item():.6f}",
                    "0.000000",
                    "g(∞) = 0",
                    f"{abs(g_bc.item() - 0.0):.6f}",
                ],
                [
                    "θ(∞)",
                    f"{theta_bc.item():.6f}",
                    "0.000000",
                    "θ(∞) = 0",
                    f"{abs(theta_bc.item() - 0.0):.6f}",
                ],
                [
                    "φ(∞)",
                    f"{phi_bc.item():.6f}",
                    "0.000000",
                    "φ(∞) = 0",
                    f"{abs(phi_bc.item() - 0.0):.6f}",
                ],
            ]

            # Print BC table
            for i, row_data in enumerate(bc_data):
                if i == 0:  # Header
                    print(
                        "".join(
                            f"{item:^{col_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )
                    print("-" * sum(col_widths))
                else:
                    print(
                        "".join(
                            f"{item:^{col_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )

            # ============= SUMMARY STATISTICS =============
            print(f"\n{'Condition Satisfaction Summary':^60}")
            print("-" * 60)

            # Calculate average errors
            ic_errors = [
                abs(f_ic.item() - 0.0),
                abs(fp_ic.item() - Sv * fpp_ic.item()),
                abs(g_ic.item() - (1.0 + Sv * gp_ic.item())),
                abs(theta_ic.item() - (1.0 + St * thetap_ic.item())),
                abs(phi_ic.item() - 1.0),
            ]

            bc_errors = [
                abs(f_bc.item() - 0.0),
                abs(fp_bc.item() - 0.0),
                abs(g_bc.item() - 0.0),
                abs(theta_bc.item() - 0.0),
                abs(phi_bc.item() - 0.0),
            ]

            avg_ic_error = np.mean(ic_errors)
            avg_bc_error = np.mean(bc_errors)
            max_ic_error = np.max(ic_errors)
            max_bc_error = np.max(bc_errors)

            summary_data = [
                ["Condition Type", "Avg Error", "Max Error", "Status"],
                [
                    "Initial Conditions",
                    f"{avg_ic_error:.2e}",
                    f"{max_ic_error:.2e}",
                    "Good" if max_ic_error < 1e-2 else "Check",
                ],
                [
                    "Boundary Conditions",
                    f"{avg_bc_error:.2e}",
                    f"{max_bc_error:.2e}",
                    "Good" if max_bc_error < 1e-2 else "Check",
                ],
            ]

            summary_widths = [20, 12, 12, 10]
            for i, row_data in enumerate(summary_data):
                if i == 0:  # Header
                    print(
                        "".join(
                            f"{item:^{summary_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )
                    print("-" * sum(summary_widths))
                else:
                    print(
                        "".join(
                            f"{item:^{summary_widths[j]}}"
                            for j, item in enumerate(row_data)
                        )
                    )

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("=" * 80)


def comprehensive_plotting(
    model,
    param_cols,
    train_df,
    test_df,
    history,
    train_preds_df,
    test_preds_df,
    compare_df,
    metrics_df,
):
    """
    Generate comprehensive plotting including boundary/initial condition verification.
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PLOTTING AND ANALYSIS")
    print("=" * 80)

    # -------------------------
    # PLOTS (Seaborn + Matplotlib) - Both versions generated
    # -------------------------
    print("\n=== Generating Basic Loss and Performance Plots ===")

    # 1) Combined Test + Training graph in a single plot (total loss)
    save_dual_line(
        history["epoch"],
        [history["train_total"], history["test_total"]],
        ["Train Total", "Test Total"],
        "Combined Total Loss (Train vs Test)",
        "Epoch",
        "Loss",
        "combined_total_loss",
    )

    # 2) Separate Test and Training graph (each containing separate curve for Data loss, BC, IC and PDE)
    save_dual_line(
        history["epoch"],
        [
            history["train_data"],
            history["train_bc"],
            history["train_ic"],
            history["train_pde"],
        ],
        ["Data Loss", "BC Loss", "IC Loss", "PDE Loss"],
        "Training Loss Components",
        "Epoch",
        "Loss",
        "train_loss_components",
    )

    save_dual_line(
        history["epoch"],
        [
            history["test_data"],
            history["test_bc"],
            history["test_ic"],
            history["test_pde"],
        ],
        ["Data Loss", "BC Loss", "IC Loss", "PDE Loss"],
        "Test Loss Components",
        "Epoch",
        "Loss",
        "test_loss_components",
    )

    # 3) 4 graphs each containing two curves of Test and Train for these Data loss, BC Loss, IC Loss and PDE Loss respectively.
    comps = [
        ("data", "Data Loss"),
        ("bc", "BC Loss"),
        ("ic", "IC Loss"),
        ("pde", "PDE Loss"),
    ]
    for comp_key, comp_label in comps:
        save_dual_line(
            history["epoch"],
            [history[f"train_{comp_key}"], history[f"test_{comp_key}"]],
            [f"Train {comp_label}", f"Test {comp_label}"],
            f"{comp_label} (Train vs Test)",
            "Epoch",
            "Loss",
            f"{comp_key}_train_vs_test",
        )

    # 4) Scatter plots for Cf, Nu, Sh Predicted and Actual.
    print("\n=== Generating Prediction Scatter Plots ===")
    for var in ["Cf", "Nu", "Sh"]:
        save_dual_scatter(
            compare_df[f"{var}_true"].values,
            compare_df[f"{var}_pred"].values,
            f"Scatter Predicted vs Actual: {var}",
            f"{var} Actual",
            f"{var} Predicted",
            f"scatter_{var}",
            with_line=True,
        )

    # 5) Bar plots for metrics (RMSE, MAE, MSE, R2)
    print("\n=== Generating Performance Metrics Bar Charts ===")
    save_dual_bar(
        metrics_df["Variable"].tolist(),
        [
            metrics_df["RMSE"].tolist(),
            metrics_df["MAE"].tolist(),
            metrics_df["MSE"].tolist(),
        ],
        ["RMSE", "MAE", "MSE"],
        "Error Metrics (Test)",
        "Variable",
        "metrics_bar",
    )

    # 6) 3D response surface plots for each Cf, Nu and Sh: z = variable, x = M, y = lp
    print("\n=== Generating 3D Response Surface Plots ===")

    def make_3d_mesh(df_preds, var_true_col, var_pred_col, name):
        x = df_preds["M"].values
        y = df_preds["lp"].values
        z_true = df_preds[var_true_col].values
        z_pred = df_preds[var_pred_col].values

        # Dual 3D scatter plots (both seaborn and matplotlib versions)
        save_dual_3d(
            x,
            y,
            z_true,
            f"3D Response (True) - {name}",
            f"${param_labels.get('M', 'M')}$",
            f"${param_labels.get('lp', r'\\lambda_p')}$",
            f"${name}$",
            f"3d_{name}_true",
        )
        save_dual_3d(
            x,
            y,
            z_pred,
            f"3D Response (Predicted) - {name}",
            f"${param_labels.get('M', 'M')}$",
            f"${param_labels.get('lp', r'\\lambda_p')}$",
            f"${name}$",
            f"3d_{name}_pred",
        )

        # Dual 3D surface plots
        save_dual_3d_surface(
            x,
            y,
            z_true,
            f"3D Surface (True) - {name}",
            f"${param_labels.get('M', 'M')}$",
            f"${param_labels.get('lp', r'\\lambda_p')}$",
            f"${name}$",
            f"3d_surface_{name}_true",
        )
        save_dual_3d_surface(
            x,
            y,
            z_pred,
            f"3D Surface (Predicted) - {name}",
            f"${param_labels.get('M', 'M')}$",
            f"${param_labels.get('lp', r'\\lambda_p')}$",
            f"${name}$",
            f"3d_surface_{name}_pred",
        )

    make_3d_mesh(test_preds_df, "Cf_true", "Cf_pred", "Cf")
    make_3d_mesh(test_preds_df, "Nu_true", "Nu_pred", "Nu")
    make_3d_mesh(test_preds_df, "Sh_true", "Sh_pred", "Sh")

    # 7) Combined Train+Test loss component plot (overlay train and test components in a single figure)
    print("\n=== Generating Combined Component Plots ===")
    save_dual_line(
        history["epoch"],
        [
            history["train_data"],
            history["test_data"],
            history["train_bc"],
            history["test_bc"],
            history["train_ic"],
            history["test_ic"],
            history["train_pde"],
            history["test_pde"],
        ],
        [
            "Train Data",
            "Test Data",
            "Train BC",
            "Test BC",
            "Train IC",
            "Test IC",
            "Train PDE",
            "Test PDE",
        ],
        "All Loss Components (Train & Test)",
        "Epoch",
        "Loss",
        "all_components_train_test",
    )

    # 8) Log-scale versions of all loss plots
    print("\n=== Generating Log-Scale Loss Plots ===")

    # Combined total loss (log scale)
    save_dual_line_log(
        history["epoch"],
        [history["train_total"], history["test_total"]],
        ["Train Total", "Test Total"],
        "Combined Total Loss (Train vs Test)",
        "Epoch",
        "Loss",
        "combined_total_loss",
    )

    # Training loss components (log scale)
    save_dual_line_log(
        history["epoch"],
        [
            history["train_data"],
            history["train_bc"],
            history["train_ic"],
            history["train_pde"],
        ],
        ["Data Loss", "BC Loss", "IC Loss", "PDE Loss"],
        "Training Loss Components",
        "Epoch",
        "Loss",
        "train_loss_components",
    )

    # Test loss components (log scale)
    save_dual_line_log(
        history["epoch"],
        [
            history["test_data"],
            history["test_bc"],
            history["test_ic"],
            history["test_pde"],
        ],
        ["Data Loss", "BC Loss", "IC Loss", "PDE Loss"],
        "Test Loss Components",
        "Epoch",
        "Loss",
        "test_loss_components",
    )

    # Individual component comparisons (log scale)
    comps_log = [
        ("data", "Data Loss"),
        ("bc", "BC Loss"),
        ("ic", "IC Loss"),
        ("pde", "PDE Loss"),
    ]
    for comp_key, comp_label in comps_log:
        save_dual_line_log(
            history["epoch"],
            [history[f"train_{comp_key}"], history[f"test_{comp_key}"]],
            [f"Train {comp_label}", f"Test {comp_label}"],
            f"{comp_label} (Train vs Test)",
            "Epoch",
            "Loss",
            f"{comp_key}_train_vs_test",
        )

    # 9) Loss gradient analysis
    print("\n=== Generating Loss Gradient Analysis ===")

    # Total loss gradient analysis
    save_dual_gradient_analysis(
        history["epoch"],
        [history["train_total"], history["test_total"]],
        ["Train Total", "Test Total"],
        "Total Loss",
        "Epoch",
        "total_loss_gradient",
    )

    # Component loss gradient analysis
    save_dual_gradient_analysis(
        history["epoch"],
        [
            history["train_data"],
            history["train_bc"],
            history["train_ic"],
            history["train_pde"],
        ],
        ["Data", "BC", "IC", "PDE"],
        "Training Loss Components",
        "Epoch",
        "train_components_gradient",
    )

    # 10) Performance metrics radar charts
    print("\n=== Generating Performance Radar Charts ===")
    save_dual_radar_chart(
        metrics_df, "Performance Metrics Comparison", "performance_radar"
    )

    # 11) Feature importance analysis
    print("\n=== Generating Feature Importance Analysis ===")
    save_dual_feature_importance(
        param_cols, train_df, test_df, test_preds_df, "feature_importance"
    )

    # 12) Q-Q plots for residual analysis
    print("\n=== Generating Q-Q Plots for Residual Analysis ===")
    for var in ["Cf", "Nu", "Sh"]:
        if f"{var}_true" in compare_df.columns and f"{var}_pred" in compare_df.columns:
            save_dual_qq_plots(
                compare_df[f"{var}_true"].values,
                compare_df[f"{var}_pred"].values,
                var,
                f"qq_plot_{var}",
            )

    print("\nAll figures saved. Summary files:")
    print(" - predicted_vs_true_test.csv")
    print(" - metrics_test.csv")
    print("\n=== COMPREHENSIVE PLOTTING COMPLETE ===")
    print("Generated plot types (all in BOTH Seaborn and Matplotlib versions):")
    print("Loss evolution plots (linear and log scale)")
    print("Loss gradient analysis (linear and log scale)")
    print("Scatter plots with reference lines")
    print("Performance metrics bar charts")
    print("Performance radar charts")
    print("3D scatter plots")
    print("3D surface plots")
    print("Feature importance analysis")
    print("Q-Q plots for residual analysis")
    print(
        f"\nTotal plots generated: {len([f for f in os.listdir('.') if f.endswith('.png')])} files"
    )
    print("Every visualization available in both seaborn and matplotlib styles!")

    # -------------------------
    # BOUNDARY & INITIAL CONDITION VERIFICATION
    # -------------------------
    print("\n=== Performing Boundary & Initial Condition Verification ===")
    verify_boundary_and_initial_conditions(model, param_cols, test_df)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS COMPLETE")
    print("=" * 80)

    # Call comprehensive plotting function
    comprehensive_plotting(
        model,
        param_cols,
        train_df,
        test_df,
        history,
        train_preds_df,
        test_preds_df,
        compare_df,
        metrics_df,
    )
