# src/emergent/cli.py
"""
M15-cli: Main command-line interface for the emergent suite.
"""
from __future__ import annotations

import math
import typer
from rich.console import Console
from rich.table import Table
import numpy as np

# --- Core simulation modules ---
from emergent.poset import CausalSite
from emergent.entropy import run_entropy_simulation
from emergent.chsh import find_spacelike_nodes, run_chsh_experiment
from emergent.geom import get_nodes_in_ball, estimate_mm_dimension, analyze_curvature_in_ball
from emergent.extremum import find_entropy_maximizer

# --- Phase B/D addon CLIs ---
from emergent.cli_predict import app as predict_app
from emergent.cli_entropy_max import app as entropy_max_app # CORRECTED

app = typer.Typer(name="emergent", help="Causal-site numerical suite.", add_completion=False)
console = Console()

# --- Mount the addon CLIs ---
app.add_typer(predict_app, name="predict")
app.add_typer(entropy_max_app, name="entropy-max") # CORRECTED


@app.command()
def entropy(
    layers: int = typer.Option(4, help="Depth layers."),
    nodes: int = typer.Option(3, help="Nodes per layer."),
    q: int = typer.Option(2, help="Alphabet size."),
    r: int = typer.Option(3, help="Max out-degree + 1."),
    ticks: int = typer.Option(3, help="Ticks to simulate."),
    observer_depth: int = typer.Option(2, help="Visible depth (< observer_depth)."),
    seed: int = typer.Option(1, help="RNG seed."),
):
    """Runs a quick entropy simulation using the static fiber model."""
    console.print("[bold cyan]Running Static Entropy Simulation...[/bold cyan]")
    console.print(f"Parameters: layers={layers}, nodes={nodes}, q={q}, R={r}, seed={seed}")

    rng = np.random.default_rng(seed)
    site = CausalSite.generate(n_layers=layers, nodes_per_layer=nodes, R=r, edge_prob=0.7, rng=rng)
    initial_config = {n: int(rng.integers(q)) for n in site.nodes}

    entropies, increments = run_entropy_simulation(site, q, initial_config, observer_depth, ticks)

    table = Table("Tick (t)", "Entropy S_t", "Increment ΔS_t")
    table.add_row("0", f"{entropies[0]:.4f}", "")
    for i, inc in enumerate(increments):
        table.add_row(str(i + 1), f"{entropies[i + 1]:.4f}", f"{inc:.4f}")
    console.print(table)


@app.command()
def chsh(
    layers: int = typer.Option(4, help="Depth layers."),
    nodes: int = typer.Option(3, help="Nodes per layer."),
    r: int = typer.Option(3, help="Max out-degree + 1."),
    seed: int = typer.Option(10, help="RNG seed."),
):
    """Runs a CHSH experiment and checks the classical/Tsirelson bounds."""
    # ... (rest of the chsh function is unchanged)
    q = 2  # binary outcomes
    console.print("[bold cyan]Running CHSH Experiment...[/bold cyan]")

    rng = np.random.default_rng(seed)
    site = CausalSite.generate(n_layers=layers, nodes_per_layer=nodes, R=r, edge_prob=0.8, rng=rng)

    pair = find_spacelike_nodes(site, rng)
    if not pair:
        console.print("[bold red]Error:[/bold red] Could not find spacelike nodes.")
        raise typer.Exit(1)

    alice_node, bob_node = pair
    console.print(f"Spacelike pair: Alice {alice_node}, Bob {bob_node}")

    max_depth = max(site.depths.values())
    final_nodes = [n for n, d in site.depths.items() if d == max_depth]
    if not final_nodes:
        console.print("[bold red]Error:[/bold red] Site has no final layer.")
        raise typer.Exit(1)

    observer_node = final_nodes[0]
    hidden_nodes = [n for n, d in site.depths.items() if d == 0]

    max_s = 0.0
    for tag in range(q):
        s_val = run_chsh_experiment(
            site=site,
            q=q,
            observer_config={observer_node: tag},
            alice_node=alice_node,
            bob_node=bob_node,
            hidden_source_nodes=hidden_nodes,
        )
        max_s = max(max_s, abs(s_val))

    console.print(f"\nMax observed S-value: [bold green]{max_s:.4f}[/bold green]")
    console.print(f"Classical Bound:      2.0000")
    console.print(f"Tsirelson Bound:      {2*np.sqrt(2):.4f}")
    if max_s > 2 * math.sqrt(2) + 1e-9:
        console.print("[bold red]VIOLATION DETECTED![/bold red]")
    else:
        console.print("[green]Bound respected.[/green]")


@app.command()
def lorentz(
    layers: int = typer.Option(12, help="Depth layers."),
    nodes: int = typer.Option(20, help="Nodes per layer."),
    r: int = typer.Option(5, help="Max out-degree + 1."),
    seed: int = typer.Option(42, help="RNG seed."),
):
    """Calculates geometric diagnostics (dimension/curvature) at different scales."""
    # ... (rest of the lorentz function is unchanged)
    console.print("[bold cyan]Running Geometric Diagnostics...[/bold cyan]")

    rng = np.random.default_rng(seed)
    site = CausalSite.generate(n_layers=layers, nodes_per_layer=nodes, R=r, edge_prob=0.5, rng=rng)
    center = site.nodes[len(site.nodes) // 2]
    radii = [3.0, 4.0, 5.0, 6.0]

    table = Table("Radius", "Ball Size", "MM-Dimension", "Mean Curvature", "Var Curvature")
    for rad in radii:
        ball = get_nodes_in_ball(site, center, rad)
        if len(ball) < 15:
            continue
        dim = estimate_mm_dimension(site, ball)
        mean_k, var_k = analyze_curvature_in_ball(site, ball)
        table.add_row(f"{rad:.1f}", str(len(ball)), f"{dim:.4f}", f"{mean_k:.4f}", f"{var_k:.4f}")

    console.print(table)


@app.command()
def extremum(
    qmin: int = typer.Option(2, help="Min q"), qmax: int = typer.Option(12, help="Max q"),
    rmin: int = typer.Option(1, help="Min R"), rmax: int = typer.Option(8, help="Max R"),
):
    """Finds the (q, R) pair that maximizes the phenomenological entropy density."""
    # ... (rest of the extremum function is unchanged)
    console.print("[bold cyan]Searching for Entropy Maximizer (q*, R*)...[/bold cyan]")
    (q_star, r_star), max_s = find_entropy_maximizer((qmin, qmax), (rmin, rmax))
    console.print(f"\nSearch complete over q=[{qmin},{qmax}], R=[{rmin},{rmax}].")
    console.print(f"Unique maximum found at: [bold green](q*, R*) = ({q_star}, {r_star})[/bold green]")
    console.print(f"Maximum entropy density S_inf: [bold green]{max_s:.4f}[/bold green]")


if __name__ == "__main__":
    app()