# src/emergent/cli_entropy_max.py
"""
Phase B CLI for the entropy maximizer (M14).
"""
from __future__ import annotations
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table

from emergent.entropy_max import EntropyMaxConfig, anneal_qR

app = typer.Typer(
    no_args_is_help=True, help="Discrete annealing over (q,R) to find the entropy maximizer."
)

@app.command("run")
def cli_entropy_max(
    qmin: int = 2,
    qmax: int = 20,
    rmin: int = 1,
    rmax: int = 10,
    budget: int = 200,
    temp0: float = 0.5,
    alpha: float = 0.22,
    n_graphs: int = 4,
    n_layers: int = 12,
    nodes_per_layer: int = 24,
    edge_prob: float = 0.45,
    observer_depth: Optional[int] = None,
    seed: int = 12345,
):
    """
    Finds (q*, R*) by maximizing the measured entropy-density objective.
    """
    cfg = EntropyMaxConfig(
        q_range=(qmin, qmax),
        R_range=(rmin, rmax),
        budget=budget,
        temperature0=temp0,
        alpha_connectivity=alpha,
        n_graphs=n_graphs,
        n_layers=n_layers,
        nodes_per_layer=nodes_per_layer,
        edge_prob=edge_prob,
        observer_depth=observer_depth,
        seed=seed,
    )
    (q_star, r_star), best, trace = anneal_qR(cfg)

    con = Console()
    con.print(f"[bold]Best (q*,R*) = ({q_star},{r_star})[/bold]  score={best:.6f}")
    if trace:
        tbl = Table(title="Accepted steps (last 10)")
        tbl.add_column("q"); tbl.add_column("R"); tbl.add_column("score")
        for qv, rv, sc in trace[-10:]:
            tbl.add_row(str(qv), str(rv), f"{sc:.6f}")
        con.print(tbl)