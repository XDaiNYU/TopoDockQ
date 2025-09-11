import argparse
from prody import parsePDB, writePDB

def extract_interface_residues(pdb_file, chain1, chain2, distance_cutoff, output_file):
    """
    Extract all residues where any heavy atom is within `distance_cutoff` Å
    of the other chain.
    """

    # Load structure
    structure = parsePDB(pdb_file)

    # Select protein atoms for each chain
    sel_chain1 = structure.select(f"chain {chain1} and protein")
    sel_chain2 = structure.select(f"chain {chain2} and protein")

    if sel_chain1 is None or sel_chain2 is None:
        raise ValueError(f"Could not find chains {chain1} or {chain2} in the structure.")

    # Find all atoms in the other chain within cutoff
    near_chain1 = structure.select(
        f"protein and chain {chain2} and within {distance_cutoff} of (chain {chain1} and protein)"
    )
    near_chain2 = structure.select(
        f"protein and chain {chain1} and within {distance_cutoff} of (chain {chain2} and protein)"
    )

    if near_chain1 is None and near_chain2 is None:
        raise ValueError("No interface residues found within the specified cutoff.")

    # Collect residues (full residues, not just the close atoms)
    residues_to_keep = set()

    def add_residues(atom_group):
        if atom_group is not None:
            for resnum, chain_id in zip(atom_group.getResnums(), atom_group.getChids()):
                residues_to_keep.add((chain_id, resnum))

    add_residues(near_chain1)
    add_residues(near_chain2)

    # Create selection string to grab all atoms of these residues
    selection_parts = [f"(chain {chain} and resnum {resnum})" for chain, resnum in residues_to_keep]
    selection_str = " or ".join(selection_parts)

    subset = structure.select(selection_str)

    if subset is None:
        raise ValueError("No atoms selected for output. Check inputs.")

    # Save result
    writePDB(output_file, subset)
    print(f"✅ Saved full-residue subset PDB to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract interface residues within N Å between two chains.")
    parser.add_argument("--pdb", required=True, help="Input PDB file path")
    parser.add_argument("--chain1", required=True, help="First chain ID")
    parser.add_argument("--chain2", required=True, help="Second chain ID")
    parser.add_argument("--distance", type=float, default=10.0, help="Distance cutoff in Å (default: 10.0)")
    parser.add_argument("--out", required=True, help="Output PDB file path")

    args = parser.parse_args()
    extract_interface_residues(args.pdb, args.chain1, args.chain2, args.distance, args.out)

