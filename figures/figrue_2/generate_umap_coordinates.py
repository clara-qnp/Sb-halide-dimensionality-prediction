"""Generate the two UMAP coordinate tables consumed by Figure 2 panel (c).

The original-cation embedding is fitted on RDKit descriptors from compounds with
known dimensionality. Newly proposed cations are transformed into that fitted space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
import umap


DIMS = {"0D", "1D", "2D", "3D"}
RANDOM_STATE = 37


def canonicalize(smiles: str) -> str | None:
    if not isinstance(smiles, str):
        return None
    molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(molecule) if molecule is not None else None


def calculate_descriptors(smiles: str) -> dict[str, float | bool]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {
            "smiles_valid": False,
            **{name: np.nan for name, _ in Descriptors.descList},
        }
    values = Descriptors.CalcMolDescriptors(
        molecule,
        missingVal=np.nan,
        silent=False,
    )
    return {"smiles_valid": True, **values}


def load_original_cations(path: Path) -> pd.DataFrame:
    compounds = pd.read_excel(path)
    required = {"dimensionality", "canonical_cation_smiles"}
    missing = required - set(compounds.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    compounds["dimensionality"] = compounds["dimensionality"].astype(str).str.strip()
    compounds = compounds[compounds["dimensionality"].isin(DIMS)].copy()
    compounds["canonical_cation_smiles"] = compounds["canonical_cation_smiles"].map(canonicalize)
    return (
        compounds[["canonical_cation_smiles"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"canonical_cation_smiles": "canonical_unique_cation"})
        .reset_index(drop=True)
    )


def load_new_cations(path: Path) -> pd.DataFrame:
    new_cations = pd.read_csv(path)
    required = {"cation_name", "SMILES"}
    missing = required - set(new_cations.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    new_cations = new_cations[["cation_name", "SMILES"]].copy()
    new_cations["SMILES"] = new_cations["SMILES"].map(canonicalize)
    if new_cations["SMILES"].isna().any():
        invalid = new_cations.loc[new_cations["SMILES"].isna(), "cation_name"].tolist()
        raise ValueError(f"Invalid new-cation SMILES for: {invalid}")
    return new_cations


def fit_and_project(
    original_cations: pd.DataFrame,
    new_cations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = pd.concat(
        [
            original_cations,
            pd.DataFrame(
                original_cations["canonical_unique_cation"]
                .map(calculate_descriptors)
                .tolist()
            ),
        ],
        axis=1,
    )
    original = original[original["smiles_valid"]].reset_index(drop=True)

    descriptor_names = {name for name, _ in Descriptors.descList}
    descriptor_columns = [column for column in original.columns if column in descriptor_names]
    original_x = original[descriptor_columns]
    original_x = original_x.loc[:, original_x.nunique(dropna=False) > 1]

    scaler = StandardScaler()
    original_scaled = scaler.fit_transform(original_x)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=RANDOM_STATE,
    )
    original[["UMAP_1", "UMAP_2"]] = reducer.fit_transform(original_scaled)

    new_cations = new_cations.copy()
    new_cations["in_original"] = new_cations["SMILES"].isin(
        original["canonical_unique_cation"]
    )
    new_descriptor_table = pd.DataFrame(
        new_cations["SMILES"].map(calculate_descriptors).tolist()
    )
    projected = pd.concat([new_cations, new_descriptor_table], axis=1)
    new_scaled = scaler.transform(projected[original_x.columns])
    new_coordinates = reducer.transform(new_scaled)

    # The overlapping compounds anchor a possible reflection of transformed points.
    overlap_mask = projected["in_original"].to_numpy()
    if overlap_mask.sum() >= 2:
        fitted_u1 = (
            original.set_index("canonical_unique_cation")
            .loc[projected.loc[overlap_mask, "SMILES"], "UMAP_1"]
            .to_numpy()
        )
        transformed_u1 = new_coordinates[overlap_mask, 0]
        if (transformed_u1[1] - transformed_u1[0]) * (fitted_u1[1] - fitted_u1[0]) < 0:
            new_coordinates[:, 0] *= -1

    projected[["UMAP_1", "UMAP_2"]] = new_coordinates

    # Plot overlapping cations at their exact coordinates in the fitted embedding.
    overlap_order = projected.loc[projected["in_original"], "SMILES"]
    fitted_overlap = (
        original.set_index("canonical_unique_cation")
        .loc[overlap_order, ["UMAP_1", "UMAP_2"]]
    )
    projected.loc[projected["in_original"], ["UMAP_1", "UMAP_2"]] = (
        fitted_overlap.to_numpy()
    )
    return original, projected


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=repository_root / "data" / "merged_compound_cif_cation.xlsx",
        help="Curated compound workbook containing dimensionality and cation SMILES.",
    )
    parser.add_argument(
        "--new-cations",
        type=Path,
        default=script_dir / "new_cations.csv",
        help="CSV containing cation_name and SMILES for the proposed cations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "generated_coordinates",
        help="Directory for the two generated coordinate CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original_cations = load_original_cations(args.data)
    new_cations = load_new_cations(args.new_cations)
    original, projected = fit_and_project(original_cations, new_cations)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    original_path = args.output_dir / "umap_original_cations.csv"
    new_path = args.output_dir / "umap_new_cations.csv"
    original[["canonical_unique_cation", "UMAP_1", "UMAP_2"]].to_csv(
        original_path,
        index=False,
    )
    projected[["cation_name", "SMILES", "in_original", "UMAP_1", "UMAP_2"]].to_csv(
        new_path,
        index=False,
    )
    print(f"Saved {len(original)} original-cation coordinates to {original_path}")
    print(f"Saved {len(projected)} proposed-cation coordinates to {new_path}")


if __name__ == "__main__":
    main()
