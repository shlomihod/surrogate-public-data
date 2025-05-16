import pandas as pd
import json
import hashlib
import os
import requests
from urllib.parse import urlparse
import subprocess


def is_url(path):
    p = urlparse(path)
    return p.scheme in ("http", "https")

def load_dataframe(path):
    return pd.read_csv(path) if is_url(path) else pd.read_csv(path)

def load_metadata(path):
    if is_url(path):
        resp = requests.get(path)
        resp.raise_for_status()
        return resp.json()
    else:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def compute_sha256(path):
    if is_url(path):
        resp = requests.get(path)
        resp.raise_for_status()
        data = resp.content
    else:
        with open(path, "rb") as f:
            data = f.read()
    return hashlib.sha256(data).hexdigest()


def generate_croissant_jsonl(data_csv_path, metadata_json_path, output_jsonl_path):
    df = load_dataframe(data_csv_path)
    metadata = load_metadata(metadata_json_path)

    dataset = {
    "@context": {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "sc":    "https://schema.org/",
        "cr":    "http://mlcommons.org/croissant/",
        "rai":   "http://mlcommons.org/croissant/RAI/",
        "dct":   "http://purl.org/dc/terms/",
        "citeAs":"cr:citeAs",
        "column":"cr:column",
        "conformsTo": "dct:conformsTo",
        "data": {
        "@id":   "cr:data",
        "@type": "@json"
        },
        "dataType": {
        "@id":   "cr:dataType",
        "@type": "@vocab"
        },
        "examples": {
        "@id":   "cr:examples",
        "@type": "@json"
        },
        "extract":     "cr:extract",
        "field":       "cr:field",
        "fileProperty":"cr:fileProperty",
        "fileObject":  "cr:fileObject",
        "fileSet":     "cr:fileSet",
        "format":      "cr:format",
        "includes":    "cr:includes",
        "isLiveDataset":"cr:isLiveDataset",
        "jsonPath":    "cr:jsonPath",
        "key":         "cr:key",
        "md5":         "cr:md5",
        "parentField": "cr:parentField",
        "path":        "cr:path",
        "recordSet":   "cr:recordSet",
        "references":  "cr:references",
        "regex":       "cr:regex",
        "repeated":    "cr:repeated",
        "replace":     "cr:replace",
        "separator":   "cr:separator",
        "source":      "cr:source",
        "subField":    "cr:subField",
        "transform":   "cr:transform"
    },
    "@type": "sc:Dataset"
    }

    # mandatory: name
    if "name" in metadata:
        dataset["name"] = metadata["name"]
    else:
        dataset["name"] = os.path.splitext(os.path.basename(data_csv_path))[0]

    for key in ("description", "license", "url", "creator", "datePublished", "version"):
        if key in metadata:
            dataset[key] = metadata[key]

    dataset["citation"] = metadata.get("citation",
        "Hod, Shlomi; Rosenblatt, Lucas; Stoyanovich, Julia. "
        "Do You Really Need Public Data? Surrogate Public Data for Differential Privacy on Tabular Data. "
        "arXiv preprint arXiv:2504.14368, 2025."
    )
    dataset["datePublished"] = metadata.get("datePublished", "2025")
    dataset["license"] = metadata.get("license",
        "https://opensource.org/licenses/MIT"
    )
    dataset["version"] = metadata.get("version", "0.1")

    filename = os.path.basename(urlparse(data_csv_path).path)
    sha256 = compute_sha256(data_csv_path)
    dataset["distribution"] = [{
        "@type":        "sc:FileObject",
        "@id":          filename,
        "name":         filename,
        "contentUrl":   data_csv_path,
        "encodingFormat":"text/csv",
        "sha256":       sha256
    }]

    recordset = {
        "@type":       "cr:RecordSet",
        "name":        metadata.get("recordSetName", "records"),
        "description": metadata.get("recordSetDescription", "Columns from tabular data."),
        "field":       []
    }
    schema = metadata.get("schema", {})

    def map_dtype(dt):
        s = str(dt)
        if s.startswith("int"):   return "sc:Integer"
        if s.startswith("float"): return "sc:Float"
        return "sc:Text"

    for col in df.columns:
        m = schema.get(col)
        if not m: continue
        recordset["field"].append({
            "@type":     "cr:Field",
            "name":      col,
            "description": m.get("description",""),
            "dataType": map_dtype(m.get("dtype","")),
            "source": {
                "fileObject": {"@id": filename},
                "extract":    {"column": col}
            }
        })

    dataset["recordSet"] = [recordset]

    with open(output_jsonl_path, "w", encoding="utf-8") as fout:
        json.dump(dataset, fout, ensure_ascii=False, indent=2, sort_keys=True)
        fout.write("\n")


if __name__ == "__main__":
    base_url = "https://raw.githubusercontent.com/shlomihod/YDNPD/refs/heads/main/ydnpd/datasets/data"
    for ds in ["acs", "we", "edad"]:
        for baseline in  ["baseline_domain", "baseline_univariate", "arbitrary"]:
            data_csv      = f"{base_url}/{ds}/{baseline}.csv"
            metadata_json = f"{base_url}/{ds}/metadata.json"
            output_file   = f"{ds}_{baseline}_croissant.json"
            generate_croissant_jsonl(data_csv, metadata_json, output_file)
            print(f"gen {output_file}")
            try:
                res = subprocess.run(
                    ["mlcroissant", "validate", "--jsonld", output_file],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"[OK]   {ds}:")
                print(res.stdout)
            except subprocess.CalledProcessError as e:
                print(f"[ERR]  {ds} ({output_file}) failed validation:")
                print(e.stdout)
                print(e.stderr)
