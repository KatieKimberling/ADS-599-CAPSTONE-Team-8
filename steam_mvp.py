import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Lipinski, rdMolDescriptors, AllChem, DataStructs
import pandas as pd
from src.model_utils import load_model_assets, generate_valid_analog

st.set_page_config(
    page_title="Antiviral Molecular Generator",
    layout="wide",
)

@st.cache_resource
def get_model_assets():
    return load_model_assets()

assets = get_model_assets()

# -----------------------------
# Core helpers
# -----------------------------
def smiles_to_mol(smiles: str):
    """Convert a SMILES string into an RDKit molecule."""
    if not smiles or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles.strip())


@st.cache_data(show_spinner=False)
def mol_to_properties(smiles: str) -> dict | None:
    """Calculate a small set of readable molecular properties."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Descriptors.MolLogP(mol), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "H Bond Donors": Lipinski.NumHDonors(mol),
        "H Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Ring Count": Lipinski.RingCount(mol),
        "Heavy Atom Count": Lipinski.HeavyAtomCount(mol),
    }


@st.cache_data(show_spinner=False)
def smiles_to_image(smiles: str, size: tuple[int, int] = (420, 280)):
    """Render a molecule image from SMILES."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)

# Compute Tanimoto similarity between two SMILES strings
def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float | None:
    """Compute Tanimoto similarity between two molecules."""
    mol_a = smiles_to_mol(smiles_a)
    mol_b = smiles_to_mol(smiles_b)

    if mol_a is None or mol_b is None:
        return None

    generator = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    fp_a = generator.GetFingerprint(mol_a)
    fp_b = generator.GetFingerprint(mol_b)
    return round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 3)

# Simple Rule of Five style summary
def compare_properties(original_props: dict, generated_props: dict) -> list[str]:
    """Generate simple plain language change notes."""
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
            "The molecular weight changed from "
            f"{original_props['Molecular Weight']} to {generated_props['Molecular Weight']}."
        )

    if original_props["H Bond Donors"] != generated_props["H Bond Donors"]:
        changes.append(
            "The number of hydrogen bond donors changed from "
            f"{original_props['H Bond Donors']} to {generated_props['H Bond Donors']}."
        )

    if original_props["H Bond Acceptors"] != generated_props["H Bond Acceptors"]:
        changes.append(
            "The number of hydrogen bond acceptors changed from "
            f"{original_props['H Bond Acceptors']} to {generated_props['H Bond Acceptors']}."
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

# Simple Rule of Five style summary
def lipinski_summary(smiles: str) -> dict | None:
    """Simple Rule of Five style summary."""
    props = mol_to_properties(smiles)
    if props is None:
        return None

    passes = {
        "MW <= 500": props["Molecular Weight"] <= 500,
        "LogP <= 5": props["LogP"] <= 5,
        "HBD <= 5": props["H Bond Donors"] <= 5,
        "HBA <= 10": props["H Bond Acceptors"] <= 10,
    }

    return {
        "Pass Count": sum(passes.values()),
        "Total Rules": len(passes),
        "Details": passes,
    }


# -----------------------------
# App layout
# -----------------------------
st.title("Antiviral Molecular Generator")
st.caption(
    "Enter a SMILES string to visualize the original molecule, compare it with a generated analog, and review basic structural changes."
)

with st.container(border=True):
    st.subheader("Input")
    default_smiles = "CCO"
    input_smiles = st.text_input(
        "Input SMILES String",
        value=default_smiles,
        help="Paste a valid SMILES string here.",
    )

    generate_clicked = st.button("Generate Analog", type="primary")


if generate_clicked:
    original_mol = smiles_to_mol(input_smiles)

    if original_mol is None:
        st.error("The input SMILES string is invalid. Please enter a valid molecule.")
    else:
        generated_smiles = generate_valid_analog(assets, attempts=100, temperature=0.45)

        if generated_smiles is None:
            st.error("The model did not produce a valid SMILES string. Please try again.")
        else:
            generated_mol = smiles_to_mol(generated_smiles)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Molecule")
                st.code(input_smiles, language="text")
                original_img = smiles_to_image(input_smiles)
                if original_img is not None:
                    st.image(original_img)

            with col2:
                st.subheader("Generated Analog")
                st.code(generated_smiles, language="text")
                if generated_mol is not None:
                    generated_img = smiles_to_image(generated_smiles)
                    if generated_img is not None:
                        st.image(generated_img)
                else:
                    st.warning("The generated analog is not a valid SMILES string.")

            if generated_mol is not None:
                st.markdown("---")
                st.subheader("What Changed?")

                original_props = mol_to_properties(input_smiles)
                generated_props = mol_to_properties(generated_smiles)
                similarity = tanimoto_similarity(input_smiles, generated_smiles)
                original_lipinski = lipinski_summary(input_smiles)
                generated_lipinski = lipinski_summary(generated_smiles)

                if similarity is not None:
                    st.write(f"**Tanimoto similarity:** {similarity}")

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
                st.subheader("Lipinski Summary")

                lip_col1, lip_col2 = st.columns(2)
                with lip_col1:
                    st.write("**Original Molecule**")
                    st.write(
                        f"Passed {original_lipinski['Pass Count']} of {original_lipinski['Total Rules']} Lipinski checks."
                    )
                    st.json(original_lipinski["Details"])

                with lip_col2:
                    st.write("**Generated Analog**")
                    st.write(
                        f"Passed {generated_lipinski['Pass Count']} of {generated_lipinski['Total Rules']} Lipinski checks."
                    )
                    st.json(generated_lipinski["Details"])


with st.expander("Model status"):
    st.write(
        "This app is now connected to the saved tuned LSTM model artifacts and attempts to generate a valid SMILES string for comparison against the user input."
    )