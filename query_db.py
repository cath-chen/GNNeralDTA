import requests
import json
import os

from sympy.utilities.exceptions import sympy_deprecation_warning


def download_pdb_by_sequence(name, sequence):
    """
    Searches the PDB for structures with a 100% sequence match.
    """
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json="
    query = {
        "query": {
            "type": "terminal",
            "service": "sequence",
            "parameters": {
                "evalue_cutoff": 10 ** -100,
                "identity_cutoff": 1.0,
                "target": "pdb_protein_sequence",
                "value": sequence
            }
        },
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental", "computational"]}
    }
    response = requests.post(url, json=query)

    if response.status_code == 200:
        results = response.json()
        pdb_id = [entry['identifier'] for entry in results.get('result_set', [])][0]
    else:
        print(f"Failed to search PDB. HTTP Status: {response.status_code}")
        with open("not_found.txt", "a") as myfile:
            myfile.write(f"{name}: {sequence}\n")
        return

    if pdb_id.startswith("AF_AF"):
        cif_url = f"https://alphafold.ebi.ac.uk/files/AF-{pdb_id[5:-2]}-F1-model_v4.cif"
    else:
        cif_url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(cif_url)
    if response.status_code == 200:
        with open(f"pdb_files/{name}.cif", "wb") as f:
            f.write(response.content)
        print(f"Downloaded {name}.cif")
        return
    else:
        print(f"Failed to download CIF file for {pdb_id}. HTTP Status: {response.status_code}")


if __name__ == "__main__":
    with open('data/davis/proteins.txt') as f:
        data = f.read()

    # reconstructing the data as a dictionary
    js = json.loads(data)

    for name, sequence in js.items():
        if not os.path.exists(f"pdb_files/{name}.cif"):
            download_pdb_by_sequence(name, sequence)
        else:
            print(f"Skipping {name}")