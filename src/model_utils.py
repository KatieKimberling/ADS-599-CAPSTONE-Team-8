import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Crippen, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

# Silence noisy RDKit parse warnings in terminal
RDLogger.DisableLog("rdApp.error")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = PROJECT_ROOT / "outputs" / "artifacts"
APP_RUN_DIR = PROJECT_ROOT / "outputs" / "app_runs"
APP_RUN_DIR.mkdir(parents=True, exist_ok=True)


class SmilesLSTMTuned(torch.nn.Module):
    def __init__(self, vocab_size, pad_idx, emb_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = torch.nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        logits = self.fc(out)
        return logits


def load_model_assets(
    model_path=ARTIFACT_DIR / "smiles_lstm_tuned.pt",
    meta_path=ARTIFACT_DIR / "smiles_lstm_tuned_meta.json",
):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}
    max_len = meta["max_len"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmilesLSTMTuned(
        vocab_size=len(stoi),
        pad_idx=stoi[meta["pad_token"]],
        emb_dim=meta["emb_dim"],
        hidden_dim=meta["hidden_dim"],
        num_layers=meta["num_layers"],
        dropout=meta["dropout"],
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return {
        "model": model,
        "stoi": stoi,
        "itos": itos,
        "max_len": max_len,
        "device": device,
        "start_token": meta["start_token"],
        "end_token": meta["end_token"],
        "pad_token": meta["pad_token"],
    }


def sample_smiles_tuned(assets, temperature=0.35):
    model = assets["model"]
    stoi = assets["stoi"]
    itos = assets["itos"]
    max_len = assets["max_len"]
    device = assets["device"]
    start_token = assets["start_token"]
    end_token = assets["end_token"]
    pad_token = assets["pad_token"]

    current = torch.tensor([[stoi[start_token]]], dtype=torch.long).to(device)
    generated = []
    hidden = None

    with torch.no_grad():
        for _ in range(max_len):
            emb = model.embedding(current[:, -1:])
            out, hidden = model.lstm(emb, hidden)
            logits = model.fc(out[:, -1, :]) / temperature
            probs = torch.softmax(logits, dim=-1)

            next_idx = torch.multinomial(probs, num_samples=1).item()
            next_char = itos[next_idx]

            if next_char == end_token:
                break
            if next_char != pad_token:
                generated.append(next_char)

            current = torch.cat(
                [current, torch.tensor([[next_idx]], dtype=torch.long).to(device)],
                dim=1,
            )

    return "".join(generated)


def canonical_if_valid(smiles: str):
    if not smiles or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, canonical=True)


def molecule_properties(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "Formula": rdMolDescriptors.CalcMolFormula(mol),
        "Molecular Weight": round(Descriptors.MolWt(mol), 2),
        "LogP": round(Crippen.MolLogP(mol), 2),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "H Bond Donors": Lipinski.NumHDonors(mol),
        "H Bond Acceptors": Lipinski.NumHAcceptors(mol),
        "Ring Count": Lipinski.RingCount(mol),
        "Heavy Atom Count": Lipinski.HeavyAtomCount(mol),
    }


_fpgen = GetMorganGenerator(radius=2, fpSize=2048)


def tanimoto_similarity(smiles_a: str, smiles_b: str):
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None

    fp_a = _fpgen.GetFingerprint(mol_a)
    fp_b = _fpgen.GetFingerprint(mol_b)
    return round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 3)


def lipinski_summary(smiles: str):
    props = molecule_properties(smiles)
    if props is None:
        return None

    details = {
        "MW <= 500": props["Molecular Weight"] <= 500,
        "LogP <= 5": props["LogP"] <= 5,
        "HBD <= 5": props["H Bond Donors"] <= 5,
        "HBA <= 10": props["H Bond Acceptors"] <= 10,
    }

    return {
        "Pass Count": sum(details.values()),
        "Total Rules": len(details),
        "Details": details,
    }


def score_candidate(seed_smiles: str, candidate_smiles: str):
    sim = tanimoto_similarity(seed_smiles, candidate_smiles)
    lip = lipinski_summary(candidate_smiles)
    props = molecule_properties(candidate_smiles)

    if sim is None or lip is None or props is None:
        return None

    lip_score = lip["Pass Count"] / lip["Total Rules"]

    mw_bonus = 1.0
    if props["Molecular Weight"] < 80:
        mw_bonus = 0.85
    elif props["Molecular Weight"] > 650:
        mw_bonus = 0.80

    total_score = round(((sim * 0.7) + (lip_score * 0.3)) * mw_bonus, 4)

    return {
        "candidate_smiles": candidate_smiles,
        "similarity": sim,
        "lipinski_pass_count": lip["Pass Count"],
        "lipinski_total": lip["Total Rules"],
        "score": total_score,
        "properties": props,
        "lipinski": lip,
    }


def generate_candidate_pool(
    assets,
    seed_smiles: str,
    pool_size=8,
    max_attempts=120,
    temperature=0.35,
    min_length=5,
):
    valid_candidates = []
    seen = set()
    attempts = 0

    while len(valid_candidates) < pool_size and attempts < max_attempts:
        attempts += 1
        candidate = sample_smiles_tuned(assets, temperature=temperature)

        if not candidate or len(candidate.strip()) < min_length:
            continue

        valid = canonical_if_valid(candidate)
        if valid is None or valid in seen:
            continue

        scored = score_candidate(seed_smiles, valid)
        if scored is None:
            continue

        seen.add(valid)
        valid_candidates.append(scored)

    valid_candidates = sorted(valid_candidates, key=lambda x: x["score"], reverse=True)
    return valid_candidates


def build_selection_note(best_candidate: dict):
    sim = best_candidate["similarity"]
    lip_pass = best_candidate["lipinski_pass_count"]
    lip_total = best_candidate["lipinski_total"]

    similarity_text = "low"
    if sim >= 0.75:
        similarity_text = "high"
    elif sim >= 0.45:
        similarity_text = "moderate"

    return (
        f"This candidate was selected as the best overall valid molecule from the generated pool "
        f"because it achieved the highest combined score, with {similarity_text} structural similarity "
        f"to the input and {lip_pass} of {lip_total} Lipinski checks passed."
    )


def build_comparison_blurb(seed_smiles: str, best_candidate: dict):
    original = molecule_properties(seed_smiles)
    generated = best_candidate["properties"]

    if original is None or generated is None:
        return "A valid candidate was generated and selected for comparison."

    parts = []

    mw_diff = generated["Molecular Weight"] - original["Molecular Weight"]
    if abs(mw_diff) >= 1:
        direction = "higher" if mw_diff > 0 else "lower"
        parts.append(f"{direction} molecular weight")

    tpsa_diff = generated["TPSA"] - original["TPSA"]
    if abs(tpsa_diff) >= 1:
        direction = "higher" if tpsa_diff > 0 else "lower"
        parts.append(f"{direction} TPSA")

    logp_diff = generated["LogP"] - original["LogP"]
    if abs(logp_diff) >= 0.1:
        direction = "higher" if logp_diff > 0 else "lower"
        parts.append(f"{direction} LogP")

    if not parts:
        summary = "similar overall physicochemical properties"
    else:
        summary = ", ".join(parts)

    return (
        f"Compared with the original molecule, the selected candidate shows {summary}. "
        f"It remains chemically valid and passed {best_candidate['lipinski_pass_count']} of "
        f"{best_candidate['lipinski_total']} Lipinski checks."
    )


def save_run_log(seed_smiles: str, best_candidate: dict):
    log_path = APP_RUN_DIR / "generation_log.csv"

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_smiles": seed_smiles,
        "best_candidate_smiles": best_candidate["candidate_smiles"],
        "similarity": best_candidate["similarity"],
        "lipinski_pass_count": best_candidate["lipinski_pass_count"],
        "lipinski_total": best_candidate["lipinski_total"],
        "score": best_candidate["score"],
    }

    df_new = pd.DataFrame([row])

    if log_path.exists():
        df_existing = pd.read_csv(log_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(log_path, index=False)
    return log_path