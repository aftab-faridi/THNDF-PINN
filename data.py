import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

# ======================================================
# Global plotting configuration: Times New Roman + LaTeX
# ======================================================
plt.rcParams.update(
    {"text.usetex": False, "font.family": "Times New Roman", "font.size": 12}
)

# ======================================================
# Parameter LaTeX labels
# ======================================================
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


# ======================================================
# Compute B1..B6 from ternary hybrid nanofluid model
# ======================================================
def compute_B_params(phi_cu, phi_sic, phi_tio2):
    # Base fluid (Engine Oil) properties
    rho_f, cp_f, k_f, _mu_f, sigma_f = 884, 1910, 0.144, 0.081, 2e-4
    rhocp_f = rho_f * cp_f

    # Nanoparticles: Cu, SiC, TiO2 - Fixed TiO2 properties from Table 1
    rho_cu, cp_cu, k_cu, sigma_cu = 8933, 385, 401, 5.96e7
    rho_sic, cp_sic, k_sic, sigma_sic = 3370, 1340, 150, 140
    rho_ti, cp_ti, k_ti, sigma_ti = 4230, 692, 8.4, 1e-12

    # Total volume fraction
    _phi_t = phi_cu + phi_sic + phi_tio2

    # B1 = mu_f / mu_thnf (reciprocal of viscosity ratio from image)
    B1 = (1 - phi_cu) ** 2.5 * (1 - phi_sic) ** 2.5 * (1 - phi_tio2) ** 2.5

    # B2 = rho_thnf / rho_f (density ratio)
    B2 = (
        (1 - phi_cu) * (1 - phi_sic) * (1 - phi_tio2)
        + (1 - phi_sic) * (1 - phi_tio2) * phi_cu * rho_cu / rho_f
        + (1 - phi_tio2) * phi_sic * rho_sic / rho_f
        + phi_tio2 * rho_ti / rho_f
    )

    # B3 = (rho_cp)_thnf / (rho_cp)_f (heat capacity ratio)
    B3 = (
        (1 - phi_cu) * (1 - phi_sic) * (1 - phi_tio2)
        + (1 - phi_sic) * (1 - phi_tio2) * phi_cu * (rho_cu * cp_cu) / rhocp_f
        + (1 - phi_tio2) * phi_sic * (rho_sic * cp_sic) / rhocp_f
        + phi_tio2 * (rho_ti * cp_ti) / rhocp_f
    )

    # B4 = k_thnf / k_f (thermal conductivity ratio using Maxwell-Garnett sequential)
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

    # B5 = sigma_thnf / sigma_f (electrical conductivity ratio)
    sigma_thnf = sigma_f
    for phi, sigma in [(phi_cu, sigma_cu), (phi_sic, sigma_sic), (phi_tio2, sigma_ti)]:
        sigma_thnf = (
            sigma_thnf
            * (sigma + 2 * sigma_thnf - 2 * phi * (sigma_thnf - sigma))
            / (sigma + 2 * sigma_thnf + phi * (sigma_thnf - sigma))
        )
    B5 = sigma_thnf / sigma_f

    # B6 = D_f / D_thnf (mass diffusivity ratio)
    B6 = 1.0 / ((1 - phi_cu) * (1 - phi_sic) * (1 - phi_tio2))

    return B1, B2, B3, B4, B5, B6


# ======================================================
# Governing ODE system - FIXED
# ======================================================
def odes(eta, y, params):
    B1, B2, B3, B4, B5, B6, Lambda, M, Nm, Bc, Pr, Rd, Ec, Sc, delta, Td, n, E, lp = (
        params
    )
    f, fp, fpp, g, gp, T, Tp, C, Cp = y

    # Equation (9) - Fixed: lp/2 instead of lp**2, correct coefficients and signs
    fppp = (
        (lp / 2 * fp)
        + ((B1 * B5 / (1 + 1 / Lambda)) * M * fp)
        + (B1 * B2 / (2 * (1 + 1 / Lambda)))
        * (fp**2 - 2 * f * fpp - g**2 - Nm * (T + Bc * C))
    )

    # Equation (10) - Fixed: lp/2 instead of lp**2, correct coefficients and signs
    gpp = (
        (lp / 2 * g)
        + ((B1 * B5 / (1 + 1 / Lambda)) * M * g)
        + (B1 * B2 / (2 * (1 + 1 / Lambda)))
        * (2 * fp * g - 2 * f * gp - Nm * (T + Bc * C))
    )

    # Equation (11) - Fixed: correct coefficient structure
    Tpp = (
        -((B3 * Pr / (B4 + (4 / 3) * Rd)) * f * Tp)
        - (1 + 1 / Lambda) * Pr * (Ec / (B1 * (B4 + 4 / 3 * Rd))) * fpp**2
    )

    # Equation (12) - This was already correct
    Cpp = B6 * Sc * (delta * C * (1 + Td * T) ** n * np.exp(-E / (1 + Td * T)) - f * Cp)

    return np.vstack((fp, fpp, fppp, gp, gpp, Tp, Tpp, Cp, Cpp))


# ======================================================
# Boundary conditions
# ======================================================
def bc(ya, yb, params, Sv, St):
    f, fp, fpp, g, gp, T, Tp, C, Cp = ya
    f_inf, fp_inf, _, g_inf, _, T_inf, _, C_inf, _ = yb
    return np.array(
        [
            f,
            fp - Sv * fpp,
            g - (1 + Sv * gp),
            T - (1 + St * Tp),
            C - 1,
            fp_inf,
            g_inf,
            T_inf,
            C_inf,
        ]
    )


# ======================================================
# Solver wrapper
# ======================================================
def solve_system(params, Sv, St, eta_max=10.0, n_points=300):
    """
    Solve the boundary-value problem defined by `odes` and `bc` using scipy.solve_bvp.
    Returns (sol, Cf, Nu, Sh) where Cf, Nu, Sh follow the LaTeX definitions:
      Cf*Re^(1/2) = (1/B1) * (1 + 1/Lambda) * sqrt( f''(0)^2 + g'(0)^2 )
      Nu*Re^(-1/2) = - B4 * (1 + 4/3 * Rd) * theta'(0)
      Sh*Re^(-1/2) = - B6 * phi'(0)
    Parameters
    ----------
    params : sequence
        [B1, B2, B3, B4, B5, B6, Lambda, M, Nm, Bc, Pr, Rd, Ec, Sc, delta, Td, n, E, lp]
    Sv, St : float
        slip parameters for the disk surface
    eta_max : float, optional
        far-field truncation (default 10)
    n_points : int, optional
        number of grid points (default 300)
    """
    eta = np.linspace(0.0, eta_max, n_points)

    # initial guess: y shape = (9, n_points)
    y_guess = np.zeros((9, eta.size))
    # sensible decaying guesses for velocity / temperature / concentration modes
    y_guess[1, :] = np.exp(-eta)  # fp
    y_guess[3, :] = np.exp(-eta)  # g
    y_guess[5, :] = np.exp(-eta)  # T
    y_guess[7, :] = np.exp(-eta)  # C

    sol = solve_bvp(
        lambda e, y: odes(e, y, params),
        lambda ya, yb: bc(ya, yb, params, Sv, St),
        eta,
        y_guess,
        verbose=0,
        max_nodes=10000,
    )

    if not sol.success:
        raise RuntimeError(
            "BVP solver failed: " + getattr(sol, "message", "no message")
        )

    # Extract required derivatives/values at eta=0
    # sol.y ordering in your odes(): [f, fp, fpp, g, gp, T, Tp, C, Cp]
    fpp0 = sol.y[2, 0]  # f''(0)
    gp0 = sol.y[4, 0]  # g'(0)
    Tp0 = sol.y[6, 0]  # theta'(0)
    Cp0 = sol.y[8, 0]  # phi'(0)

    # Map parameters to names for clarity (indexes match how you packed `params`)
    B1 = params[0]
    B4 = params[3]
    B6 = params[5]
    Lambda = params[6]
    Rd = params[11]

    # Engineering quantities consistent with LaTeX
    Cf = (1.0 / B1) * (1.0 + 1.0 / Lambda) * np.sqrt(fpp0**2 + gp0**2)
    Nu = -B4 * (1.0 + (4.0 / 3.0) * Rd) * Tp0
    Sh = -B6 * Cp0

    return sol, Cf, Nu, Sh


# ======================================================
# Parameter ranges
# ======================================================
param_ranges = {
    "Lambda": (1.0, 1.5),
    "M": (0.0, 2.0),
    "Nm": (0.1, 1.0),
    "Bc": (0.1, 2.0),
    "Pr": (6.45, 6.45),
    "Rd": (0.0, 5.0),
    "Ec": (0.01, 0.5),
    "Sc": (0.5, 5.0),
    "delta": (0.1, 1.0),
    "Td": (0.1, 1.0),
    "E": (0.2, 1.2),
    "Sv": (0.0, 1.5),
    "St": (0.0, 1.5),
    "lp": (0.0, 1.5),
}
fraction_ranges = {
    "phi_cu": (0.01, 0.03),
    "phi_sic": (0.01, 0.03),
    "phi_tio2": (0.01, 0.03),
}


# ======================================================
# Plotting function
# ======================================================
# ======================================================
# Plotting function (separate plots saved individually)
# ======================================================
def plot_parameter_effect(vary_param, values, save_prefix="plot"):
    inputs = {k: np.mean(v) for k, v in param_ranges.items()}
    phi_inputs = {k: np.mean(v) for k, v in fraction_ranges.items()}

    # Storage for all solution curves
    curves = {"fp": [], "g": [], "T": [], "C": []}
    eta_grid = None
    labels = []

    for val in values:
        inputs[vary_param] = val
        B1, B2, B3, B4, B5, B6 = compute_B_params(
            phi_inputs["phi_cu"], phi_inputs["phi_sic"], phi_inputs["phi_tio2"]
        )
        params = [
            B1,
            B2,
            B3,
            B4,
            B5,
            B6,
            inputs["Lambda"],
            inputs["M"],
            inputs["Nm"],
            inputs["Bc"],
            inputs["Pr"],
            inputs["Rd"],
            inputs["Ec"],
            inputs["Sc"],
            inputs["delta"],
            inputs["Td"],
            1,
            inputs["E"],
            inputs["lp"],
        ]
        sol, Cf, Nu, Sh = solve_system(params, inputs["Sv"], inputs["St"])
        eta = sol.x
        fp, g, T, C = sol.y[1], sol.y[3], sol.y[5], sol.y[7]

        eta_grid = eta
        curves["fp"].append(fp)
        curves["g"].append(g)
        curves["T"].append(T)
        curves["C"].append(C)
        labels.append(f"${param_labels[vary_param]}={val:.2f}$")

    # --- Plot each variable separately and save ---
    def save_plot(ycurves, title, ylabel, filename):
        plt.figure(figsize=(6, 4))
        for curve, label in zip(ycurves, labels):
            plt.plot(eta_grid, curve, label=label)
        plt.title(title)
        plt.xlabel(r"$\eta$")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()

    save_plot(
        curves["fp"],
        r"Radial velocity $f'(\eta)$",
        r"$f'(\eta)$",
        f"{save_prefix}_{vary_param}_fp.png",
    )

    save_plot(
        curves["g"],
        r"Tangential velocity $g(\eta)$",
        r"$g(\eta)$",
        f"{save_prefix}_{vary_param}_g.png",
    )

    save_plot(
        curves["T"],
        r"Temperature $T(\eta)$",
        r"$T(\eta)$",
        f"{save_prefix}_{vary_param}_T.png",
    )

    save_plot(
        curves["C"],
        r"Concentration $C(\eta)$",
        r"$C(\eta)$",
        f"{save_prefix}_{vary_param}_C.png",
    )

    print(f"Saved plots for varying {vary_param} with prefix '{save_prefix}'")


# ======================================================
# Example plots
# ======================================================
plot_parameter_effect("M", [0, 0.2, 0.4, 0.6], save_prefix="set-1")
plot_parameter_effect("M", [0.3, 0.6, 0.9, 1.2], save_prefix="set-2")
plot_parameter_effect("lp", [0.3, 0.6, 0.9, 1.2], save_prefix="set-3")
plot_parameter_effect("lp", [0.2, 0.4, 0.6, 0.8], save_prefix="set-4")
plot_parameter_effect("Nm", [0.1, 0.2, 0.3, 0.4], save_prefix="set-5")
plot_parameter_effect("Nm", [0.3, 0.6, 0.9, 1.2], save_prefix="set-6")
plot_parameter_effect("Lambda", [0.2, 0.4, 0.6, 0.8], save_prefix="set-7")
plot_parameter_effect("Bc", [0.1, 0.2, 0.3, 0.4], save_prefix="set-8")
plot_parameter_effect("Rd", [0.1, 0.2, 0.3, 0.4], save_prefix="set-9")
plot_parameter_effect("delta", [0.1, 0.15, 0.2, 0.25], save_prefix="set-10")
plot_parameter_effect("E", [0.2, 0.4, 0.6, 0.8], save_prefix="set-11")

# ======================================================
# Generate CSV
# ======================================================
results = []
n_samples = 2000
for _ in range(n_samples):
    inputs = {k: np.random.uniform(*v) for k, v in param_ranges.items()}
    phi_inputs = {k: np.random.uniform(*v) for k, v in fraction_ranges.items()}
    B1, B2, B3, B4, B5, B6 = compute_B_params(
        phi_inputs["phi_cu"], phi_inputs["phi_sic"], phi_inputs["phi_tio2"]
    )
    params = [
        B1,
        B2,
        B3,
        B4,
        B5,
        B6,
        inputs["Lambda"],
        inputs["M"],
        inputs["Nm"],
        inputs["Bc"],
        inputs["Pr"],
        inputs["Rd"],
        inputs["Ec"],
        inputs["Sc"],
        inputs["delta"],
        inputs["Td"],
        1,
        inputs["E"],
        inputs["lp"],
    ]
    try:
        _, Cf, Nu, Sh = solve_system(params, inputs["Sv"], inputs["St"])
        row = {**inputs, **phi_inputs, "Cf": Cf, "Nu": Nu, "Sh": Sh}
        results.append(row)
    except RuntimeError:
        continue

df = pd.DataFrame(results)
df.to_csv("data.csv", index=False)
print("Saved data.csv with", len(df), "rows")
print(df.head())
