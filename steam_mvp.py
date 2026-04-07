import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, rdMolDescriptors, AllChem, DataStructs
import pandas as pd

from src.model_utils import (
    load_model_assets,
    generate_candidate_pool,
    build_selection_note,
    build_comparison_blurb,
    lipinski_summary,
    molecule_properties,
    tanimoto_similarity,
    save_run_log,
)

st.set_page_config(
    page_title="Antiviral Molecular Generator",
    layout="wide",
)

@st.cache_resource
def get_model_assets():
    return load_model_assets()

assets = get_model_assets()


def smiles_to_mol(smiles: str):
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


@st.cache_data(show_spinner=False)
def smiles_to_image(smiles: str, size: tuple[int, int] = (420, 280)):
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)


def compare_properties(original_props: dict, generated_props: dict) -> list[str]:
    changes: list[str] = []

    if original_props["Formula"] != generated_props["Formula"]:
        changes.append(
            f"The molecular formula changed from {original_props['Formula']} to {generated_props['Formula']}."
        )

    if original_props["Ring Count"] != generated_props["Ring Count"]:
        changes.append(
            f"The ring count changed from {original_props['Ring Count']} to {generated_props['Ring Count']}."
        )

    if original_props["Molecular Weight"] != generated_props["Molecular Weight"]:
        changes.append(
            f"The molecular weight changed from {original_props['Molecular Weight']} to {generated_props['Molecular Weight']}."
        )

    if original_props["H Bond Donors"] != generated_props["H Bond Donors"]:
        changes.append(
            f"The number of hydrogen bond donors changed from {original_props['H Bond Donors']} to {generated_props['H Bond Donors']}."
        )

    if original_props["H Bond Acceptors"] != generated_props["H Bond Acceptors"]:
        changes.append(
            f"The number of hydrogen bond acceptors changed from {original_props['H Bond Acceptors']} to {generated_props['H Bond Acceptors']}."
        )

    if original_props["TPSA"] != generated_props["TPSA"]:
        changes.append(
            f"The TPSA changed from {original_props['TPSA']} to {generated_props['TPSA']}."
        )

    if original_props["LogP"] != generated_props["LogP"]:
        changes.append(
            f"The LogP changed from {original_props['LogP']} to {generated_props['LogP']}."
        )

    return changes


def build_lipinski_table(original_lipinski: dict, generated_lipinski: dict):
    rows = []
    for rule in original_lipinski["Details"].keys():
        rows.append(
            {
                "Rule": rule,
                "Original": "Pass" if original_lipinski["Details"][rule] else "Fail",
                "Generated": "Pass" if generated_lipinski["Details"][rule] else "Fail",
            }
        )
    return pd.DataFrame(rows)


EXAMPLES = {
    "Ethanol": "CCO",
    "Acetic Acid": "CC(=O)O",
    "Ethylamine": "CCN",
    "Isopropanol": "CC(O)C",
}

if "input_smiles_value" not in st.session_state:
    st.session_state["input_smiles_value"] = "CCO"

st.title("Antiviral Molecular Generator")
st.caption(
    "Enter a SMILES string to visualize the original molecule, generate a pool of valid candidates, rank them, and review the best overall result."
)

with st.container(border=True):
    st.subheader("Examples")
    example_cols = st.columns(len(EXAMPLES))
    for col, (label, smiles) in zip(example_cols, EXAMPLES.items()):
        with col:
            if st.button(label, key=f"example_{label}"):
                st.session_state["input_smiles_value"] = smiles

with st.container(border=True):
    st.subheader("Input")
    input_smiles = st.text_input(
        "Input SMILES String",
        value=st.session_state["input_smiles_value"],
        help="Paste a valid SMILES string here.",
        key="main_smiles_input",
    )

    option_cols = st.columns(3)
    with option_cols[0]:
        pool_size = st.slider("Valid candidates to keep", min_value=3, max_value=15, value=8, step=1)
    with option_cols[1]:
        max_attempts = st.slider("Max generation attempts", min_value=20, max_value=200, value=120, step=10)
    with option_cols[2]:
        temperature = st.slider("Sampling temperature", min_value=0.20, max_value=0.70, value=0.35, step=0.05)

    generate_clicked = st.button("Generate and Rank Candidates", type="primary")

if generate_clicked:
    st.session_state["input_smiles_value"] = input_smiles
    original_mol = smiles_to_mol(input_smiles)

    if original_mol is None:
        st.error("The input SMILES string is invalid. Please enter a valid molecule.")
    else:
        with st.spinner("Generating valid candidate pool and ranking results..."):
            candidate_pool = generate_candidate_pool(
                assets,
                seed_smiles=input_smiles,
                pool_size=pool_size,
                max_attempts=max_attempts,
                temperature=temperature,
            )

        if not candidate_pool:
            st.error(
                "No valid candidates were generated in this run. Try again, lower the temperature, or increase the attempt count."
            )
        else:
            best_candidate = candidate_pool[0]
            generated_smiles = best_candidate["candidate_smiles"]
            generated_mol = smiles_to_mol(generated_smiles)

            original_props = molecule_properties(input_smiles)
            generated_props = best_candidate["properties"]
            original_lipinski = lipinski_summary(input_smiles)
            generated_lipinski = best_candidate["lipinski"]

            log_path = save_run_log(input_smiles, best_candidate)

            st.success("Valid generated candidate selected.")
            st.info(build_selection_note(best_candidate))

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Molecule")
                st.code(input_smiles, language="text")
                original_img = smiles_to_image(input_smiles)
                if original_img is not None:
                    st.image(original_img)

            with col2:
                st.subheader("Best Overall Candidate")
                st.code(generated_smiles, language="text")
                generated_img = smiles_to_image(generated_smiles)
                if generated_img is not None:
                    st.image(generated_img)

            st.markdown("---")
            st.subheader("Why this candidate was selected")
            st.write(build_comparison_blurb(input_smiles, best_candidate))

            st.markdown("---")
            st.subheader("What Changed?")

            st.write(f"**Tanimoto similarity:** {best_candidate['similarity']}")

            changes = compare_properties(original_props, generated_props)
            if changes:
                for change in changes:
                    st.write(f"• {change}")
            else:
                st.write("No major property differences were detected in this comparison.")

            st.markdown("---")
            st.subheader("Property Comparison")

            comparison_df = pd.DataFrame(
                {
                    "Property": [str(v) for v in original_props.keys()],
                    "Original": [str(v) for v in original_props.values()],
                    "Generated": [str(v) for v in generated_props.values()],
                }
            )
            st.dataframe(comparison_df, width="stretch", hide_index=True)

            st.markdown("---")
            st.subheader("Lipinski Pass/Fail Summary")
            lipinski_df = build_lipinski_table(original_lipinski, generated_lipinski)
            st.dataframe(lipinski_df, width="stretch", hide_index=True)

            st.markdown("---")
            st.subheader("Ranked Candidate Pool")

            ranked_df = pd.DataFrame(
                [
                    {
                        "Rank": idx + 1,
                        "Candidate SMILES": c["candidate_smiles"],
                        "Similarity": c["similarity"],
                        "Lipinski Passes": f"{c['lipinski_pass_count']}/{c['lipinski_total']}",
                        "Score": c["score"],
                    }
                    for idx, c in enumerate(candidate_pool)
                ]
            )
            st.dataframe(ranked_df, width="stretch", hide_index=True)

            st.markdown("---")
            st.subheader("Saved Output")
            st.write(f"Latest run saved to: `{log_path}`")

with st.expander("Limitation note"):
    st.write(
        "The current model generates candidates from the learned antiviral chemical distribution and ranks them against the user input. "
        "Generation is not yet fully conditioned on the input seed, so the selected molecule should be interpreted as the highest scoring "
        "valid candidate from the generated pool rather than a guaranteed direct analog of the input."
    )

with st.expander("Model status"):
    st.write(
        "This app is connected to the saved tuned LSTM model artifacts, generates a pool of valid candidates per run, ranks them by similarity and Lipinski performance, and saves the top result to a log."
    )