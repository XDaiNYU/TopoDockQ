# Interface File Extraction


> ⚠️ **Please install ProDy before running this script.**

The script is built on **[ProDy](http://prody.csb.pitt.edu/)** for PDB parsing and atom selections.

You can install the ProDy in a different conda environment. Suggested ProDy version is 2.4.1.

After install ProDy, or running in a conda environment with ProDy. You can install it with `prody_environment.yml`.

```bash
python extract_interface.py \
    --pdb ./example/1b3l_crystal.pdb \
    --chain1 A \
    --chain2 B \
    --distance 10.0 \
    --out ./example/1b3l_interface.pdb
