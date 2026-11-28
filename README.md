# MIRPred

MIRPred is a framework for metal ion-binding residue prediction that integrates geometry-guided representation correction (GRC) and response-constrained expert fusion (REF). Specifically, GRC refines protein structure language model representations by incorporating coordinate-derived equivariant geometric information, while leveraging structural confidence to mitigate the effects of uncertainties in predicted structures. REF employs prediction responses induced by modality replacement to guide multiple experts in capturing modality-specific, shared, and synergistic information, with adaptive residue-level routing to facilitate effective multimodal integration.

## Step 1: Get MIRPred

Enter the MIRPred directory:

```commandline
cd MIRPred-main
```

## Step 2: Build required dependencies

Run the installation script to create and activate a Conda environment named `MIRPred`:

```commandline
source install.sh
```

## Step 3: Download third-party packages

The following packages are required to generate MIRPred features:

1. **Foldseek**<br>
   Download from [Foldseek](https://github.com/steineggerlab/foldseek), place the executable at `tools/foldseek`, and run `chmod +x tools/foldseek`.

2. **SaProt**<br>
   Download [SaProt_650M_AF2](https://huggingface.co/westlake-repl/SaProt_650M_AF2) to `weights/pretrained/SaProt_650M_AF2/`.

3. **ESM-C**<br>
   Download [ESM-C 600m](https://huggingface.co/biohub/esmc-600m-2024-12) to `weights/pretrained/esmc_600m_2024_12_v0.pth`.

The primary dataset prediction checkpoint is included at `weights/MIRPred_weight.pth`.

## Step 4: Run MIRPred

Using `./example/inputs.csv` as an example, prepare the input data in the following format:

```csv
protein_id,sequence,structure_path,metal ion,PAE_json_path
```

Here, `protein_id` is the protein identifier; `sequence` is the amino acid sequence; `structure_path` is the path to a pLDDT-annotated PDB or mmCIF structure; `metal ion` is the requested metal ion type; and `PAE_json_path` is the path to the original AlphaFold PAE JSON file (optional; leave blank if unavailable). File paths are relative to the input CSV directory or absolute.

Supported metal ion types are:

```text
Zn²⁺, Ca²⁺, Mg²⁺, Mn²⁺, Fe_cation, Cu_cation, K⁺, Ni²⁺, Na⁺,
Fe³⁺, Cu²⁺, Fe²⁺, Cd²⁺, Cu⁺, Co²⁺, Hg²⁺, Co³⁺, Li⁺
```

In the `metal ion` column, enter `ALL` to predict binding residues for all 18 supported metal ions. To select specific metal ions, enter a single name such as `Zn²⁺`, or separate multiple names with semicolons, such as `Zn²⁺;Ca²⁺;Mg²⁺`.

Run the following command to obtain MIRPred predictions:

```commandline
python scripts/predict.py ./example/inputs.csv
```

The results are saved to `./example/predictions.csv`, with a binding probability and predicted label for each residue and requested metal ions.
