import argparse
from prody import parsePDB, writePDB

def extract_interface(pdb_file, chain1, chain2, distance_cutoff, output_file):
    structure = parsePDB(pdb_file)

    sel_chain1 = structure.select(f"chain {chain1} and protein")
    sel_chain2 = structure.select(f"chain {chain2} and protein")

    if sel_chain1 is None or sel_chain2 is None:
        raise ValueError(f"Could not find chains {chain1} or {chain2} in the structure.")

    near_chain1 = structure.select(f"protein and chain {chain2} and within {distance_cutoff} of chain {chain1}")
    near_chain2 = structure.select(f"protein and chain {chain1} and within {distance_cutoff} of chain {chain2}")

    combined_indices = set()
    if near_chain1 is not None:
        combined_indices.update(near_chain1.getIndices())
    if near_chain2 is not None:
        combined_indices.update(near_chain2.getIndices())

    if not combined_indices:
        raise ValueError("No atoms found within the specified distance between the chains.")

    subset = structure.select(f"index {' '.join(map(str, combined_indices))}")
    writePDB(output_file, subset)

    print(f"✅ Saved subset PDB to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract subset of residues within N Å between two chains.")
    parser.add_argument("--pdb", required=True, help="Input PDB file path")
    parser.add_argument("--chain1", required=True, help="First chain ID")
    parser.add_argument("--chain2", required=True, help="Second chain ID")
    parser.add_argument("--distance", type=float, default=10.0, help="Distance cutoff in Å (default: 10.0)")
    parser.add_argument("--out", required=True, help="Output PDB file path")

    args = parser.parse_args()
    extract_interface(args.pdb, args.chain1, args.chain2, args.distance, args.out)

