# import pandas as pd
# import requests
# import re
# import time

# # === FILE PATHS ===
# ppi_file = r"C:\Users\ravee\Downloads\BIOGRID-ALL-4.4.243.mitab\BIOGRID-ALL-4.4.243.mitab.txt"
# mutation_file = r"C:\Users\ravee\Downloads\S4191.txt"
# drug_file = r"C:\Users\ravee\Downloads\ChG-Miner_miner-chem-gene.tsv"

# # === STEP 1: Extract gene symbols from PPI dataset ===
# def extract_ppi_gene_symbols(ppi_file):
#     print("🔍 Loading PPI dataset...")
#     df = pd.read_csv(ppi_file, sep="\t", comment="#", low_memory=False)
#     print("✅ PPI dataset loaded")
    
#     # Determine which columns contain gene identifiers
#     # Usually columns 7 and 8 in BIOGRID, but let's check column names
#     print("📊 PPI File Columns:", df.columns.tolist())
    
#     # Extract from columns 7 and 8 (usually contain gene symbols)
#     genes_a = df.iloc[:, 7].dropna().astype(str).tolist()
#     genes_b = df.iloc[:, 8].dropna().astype(str).tolist()
    
#     # Clean gene symbols (remove any version numbers and keep only the gene name)
#     clean_genes = set()
#     for gene in genes_a + genes_b:
#         # Extract gene symbol (usually first part before any delimiter)
#         match = re.search(r'([A-Za-z0-9\-]+)', gene)
#         if match:
#             clean_genes.add(match.group(1).upper())
    
#     return clean_genes

# # === STEP 2: Extract gene symbols from mutation dataset ===
# def extract_mutation_genes(mutation_file):
#     df = pd.read_csv(mutation_file, sep="\t", comment="#")
#     print("🧪 Mutation File Columns:", df.columns.tolist())
    
#     gene_symbols = set()
    
#     # Extract gene symbols from Partner1 and Partner2 columns
#     if 'Partner1' in df.columns:
#         partner1_genes = df['Partner1'].dropna().astype(str)
#         gene_symbols.update([gene.upper() for gene in partner1_genes])
    
#     if 'Partner2' in df.columns:
#         partner2_genes = df['Partner2'].dropna().astype(str)
#         gene_symbols.update([gene.upper() for gene in partner2_genes])
    
#     # Clean gene symbols
#     clean_genes = set()
#     for gene in gene_symbols:
#         match = re.search(r'([A-Za-z0-9\-]+)', gene)
#         if match:
#             clean_genes.add(match.group(1).upper())
    
#     return clean_genes

# # === STEP 3: Extract gene symbols from drug dataset ===
# def extract_drug_genes(drug_file):
#     df = pd.read_csv(drug_file, sep="\t")
#     print("💊 Drug File Columns:", df.columns.tolist())
    
#     gene_symbols = set()
    
#     # Try to find the gene column
#     gene_col = None
#     for col in df.columns:
#         if 'gene' in col.lower():
#             gene_col = col
#             break
    
#     if gene_col:
#         genes = df[gene_col].dropna().astype(str)
#         gene_symbols.update([gene.upper() for gene in genes])
#     else:
#         print("⚠️ Could not find gene column in drug dataset")
    
#     # Clean gene symbols
#     clean_genes = set()
#     for gene in gene_symbols:
#         match = re.search(r'([A-Za-z0-9\-]+)', gene)
#         if match:
#             clean_genes.add(match.group(1).upper())
    
#     return clean_genes

# # === MAIN WORKFLOW ===
# print("=== 📊 EXTRACTING GENE SYMBOLS FROM DATASETS ===")

# print("\n=== 🔍 Processing PPI Dataset ===")
# ppi_genes = extract_ppi_gene_symbols(ppi_file)
# print(f"✅ Found {len(ppi_genes)} unique gene symbols in PPI dataset")
# print("🔍 Sample PPI genes:", list(ppi_genes)[:5])

# print("\n=== 🧪 Processing Mutation Dataset ===")
# mutation_genes = extract_mutation_genes(mutation_file)
# print(f"✅ Found {len(mutation_genes)} unique gene symbols in mutation dataset")
# print("🔍 Sample mutation genes:", list(mutation_genes)[:5])

# print("\n=== 💊 Processing Drug Dataset ===")
# drug_genes = extract_drug_genes(drug_file)
# print(f"✅ Found {len(drug_genes)} unique gene symbols in drug dataset")
# print("🔍 Sample drug genes:", list(drug_genes)[:5])

# # === FIND INTERSECTIONS ===
# print("\n=== 🔄 FINDING COMMON PROTEINS ===")

# # PPI and Drug intersection
# ppi_drug_common = ppi_genes.intersection(drug_genes)
# print(f"\n✅ Common proteins between PPI and Drug datasets: {len(ppi_drug_common)}")
# print("🔍 Sample common PPI-Drug proteins:", list(ppi_drug_common)[:10])

# # PPI and Mutation intersection
# ppi_mutation_common = ppi_genes.intersection(mutation_genes)
# print(f"\n✅ Common proteins between PPI and Mutation datasets: {len(ppi_mutation_common)}")
# print("🔍 Sample common PPI-Mutation proteins:", list(ppi_mutation_common)[:10])

# # Save common proteins to files
# print("\n=== 💾 SAVING RESULTS ===")

# with open("ppi_drug_common_proteins.txt", "w") as f:
#     for protein in sorted(ppi_drug_common):
#         f.write(f"{protein}\n")
# print("💾 Common PPI-Drug proteins saved to 'ppi_drug_common_proteins.txt'")

# with open("ppi_mutation_common_proteins.txt", "w") as f:
#     for protein in sorted(ppi_mutation_common):
#         f.write(f"{protein}\n")
# print("💾 Common PPI-Mutation proteins saved to 'ppi_mutation_common_proteins.txt'")

# # Optional: All three datasets intersection
# all_common = ppi_genes.intersection(drug_genes).intersection(mutation_genes)
# print(f"\n✅ Common proteins across all three datasets: {len(all_common)}")
# if all_common:
#     print("🔍 Common proteins across all datasets:", list(all_common))
    
#     with open("all_common_proteins.txt", "w") as f:
#         for protein in sorted(all_common):
#             f.write(f"{protein}\n")
#     print("💾 Common proteins across all datasets saved to 'all_common_proteins.txt'")
import pandas as pd
import re
import csv

# === FILE PATHS ===
ppi_file = r"C:\Users\ravee\Downloads\BIOGRID-ALL-4.4.243.mitab\BIOGRID-ALL-4.4.243.mitab.txt"
mutation_file = r"C:\Users\ravee\Downloads\S4191.txt"
drug_file = r"C:\Users\ravee\Downloads\ChG-Miner_miner-chem-gene.tsv"

# === PARSING FUNCTIONS WITH ID STANDARDIZATION ===

def extract_gene_identifiers(text):
    """Extract all possible identifiers from a text string"""
    identifiers = set()
    
    # Extract UniProt IDs (like P12345)
    uniprot_matches = re.findall(r'uniprot/swiss-prot:([A-Z0-9]+)', text)
    identifiers.update(uniprot_matches)
    
    # Extract UniProt IDs directly (if they match the pattern P#####)
    direct_uniprot = re.findall(r'(?:^|[^A-Za-z0-9])([OPQE][0-9][A-Z0-9]{3}[0-9])(?:$|[^A-Za-z0-9])', text)
    identifiers.update(direct_uniprot)
    
    # Extract gene symbols (extracting from various fields)
    gene_matches = re.findall(r'entrez gene/locuslink:([A-Za-z0-9\-]+)', text) 
    identifiers.update(gene_matches)
    
    # Extract gene names within parentheses (gene name synonym)
    gene_synonym_matches = re.findall(r'([A-Za-z0-9\-]+)\(gene name synonym\)', text)
    identifiers.update(gene_synonym_matches)
    
    # Clean up and standardize
    cleaned = set()
    for identifier in identifiers:
        # Remove version numbers and keep only main ID
        clean_id = re.sub(r'\.\d+$', '', identifier.upper())
        if clean_id and len(clean_id) > 1:  # Avoid single-letter IDs that might be too generic
            cleaned.add(clean_id)
    
    return cleaned

def extract_ppi_gene_symbols(ppi_file):
    print("🔍 Loading PPI dataset...")
    
    # Read first 5 lines to check format
    with open(ppi_file, 'r') as f:
        sample_lines = [next(f) for _ in range(5)]
    print("📄 Sample PPI data (first line):", sample_lines[0][:100] + "...")
    
    # Process file
    all_identifiers = set()
    
    with open(ppi_file, 'r') as f:
        # Skip comment lines
        for line in f:
            if line.startswith('#'):
                continue
            
            # Process data line
            columns = line.strip().split('\t')
            if len(columns) >= 9:  # Ensure we have enough columns
                # Process both interactors (columns often containing gene ID info)
                for col_idx in range(2, 6):  # Usually columns 3-6 contain identifiers
                    if col_idx < len(columns):
                        identifiers = extract_gene_identifiers(columns[col_idx])
                        all_identifiers.update(identifiers)
    
    print(f"✅ Extracted {len(all_identifiers)} unique identifiers from PPI dataset")
    print("🔍 Sample PPI identifiers:", list(all_identifiers)[:5])
    return all_identifiers

def extract_mutation_genes(mutation_file):
    df = pd.read_csv(mutation_file, sep="\t", comment="#")
    print("🧪 Mutation File Columns:", df.columns.tolist())
    
    gene_symbols = set()
    
    # Extract gene symbols from Partner1 and Partner2 columns
    if 'Partner1' in df.columns:
        partner1_genes = df['Partner1'].dropna().astype(str)
        gene_symbols.update([gene.upper() for gene in partner1_genes])
    
    if 'Partner2' in df.columns:
        partner2_genes = df['Partner2'].dropna().astype(str)
        gene_symbols.update([gene.upper() for gene in partner2_genes])
    
    # Direct examination of file content
    print(f"📊 Mutation dataset - first few rows of Partner1/Partner2:")
    sample_data = df[['Partner1', 'Partner2']].head(5).values.tolist()
    for row in sample_data:
        print(f"  - {row}")
    
    # Also include any UniProt or other IDs that might be in the file
    all_identifiers = set()
    for col in df.columns:
        for value in df[col].dropna().astype(str):
            ids = extract_gene_identifiers(value)
            all_identifiers.update(ids)
    
    gene_symbols.update(all_identifiers)
    
    # Remove very short identifiers (likely not useful)
    gene_symbols = {g for g in gene_symbols if len(g) > 1}
    
    print(f"✅ Found {len(gene_symbols)} unique gene identifiers in mutation dataset")
    print("🔍 Sample mutation identifiers:", list(gene_symbols)[:5])
    return gene_symbols

def extract_drug_genes(drug_file):
    print("🔍 Loading drug dataset...")
    
    try:
        # First try tab-separated
        df = pd.read_csv(drug_file, sep="\t")
        print("💊 Drug File Columns (TSV):", df.columns.tolist())
    except:
        # If that fails, try comma-separated
        try:
            df = pd.read_csv(drug_file)
            print("💊 Drug File Columns (CSV):", df.columns.tolist())
        except:
            # If still failing, try with low_memory=False
            df = pd.read_csv(drug_file, sep="\t", low_memory=False)
            print("💊 Drug File Columns (low_memory):", df.columns.tolist())
    
    # Print sample data to debug
    print("📊 Drug dataset - first few rows:")
    sample_data = df.head(3).values.tolist()
    for row in sample_data:
        print(f"  - {row[:min(3, len(row))]}")  # Show first 3 columns or less
    
    gene_symbols = set()
    
    # Try to find the gene column
    gene_col = None
    for col in df.columns:
        if 'gene' in col.lower():
            gene_col = col
            break
    
    if gene_col:
        genes = df[gene_col].dropna().astype(str)
        print(f"📊 Found gene column: '{gene_col}' with {len(genes)} entries")
        
        # Extract all identifiers
        for gene in genes:
            # Direct identifier - just the text itself
            gene_symbols.add(gene.upper())
            
            # Try to extract embedded identifiers
            ids = extract_gene_identifiers(gene)
            gene_symbols.update(ids)
    else:
        print("⚠️ Could not find gene column in drug dataset")
        # Try to extract identifiers from all columns
        for col in df.columns:
            for value in df[col].dropna().astype(str):
                ids = extract_gene_identifiers(value)
                gene_symbols.update(ids)
    
    # Remove very short identifiers (likely not useful)
    gene_symbols = {g for g in gene_symbols if len(g) > 1}
    
    print(f"✅ Found {len(gene_symbols)} unique gene identifiers in drug dataset")
    print("🔍 Sample drug identifiers:", list(gene_symbols)[:5])
    return gene_symbols

# === MAIN WORKFLOW ===
print("=== 📊 EXTRACTING GENE SYMBOLS FROM DATASETS ===")

print("\n=== 🔍 Processing PPI Dataset ===")
ppi_genes = extract_ppi_gene_symbols(ppi_file)

print("\n=== 🧪 Processing Mutation Dataset ===")
mutation_genes = extract_mutation_genes(mutation_file)

print("\n=== 💊 Processing Drug Dataset ===")
drug_genes = extract_drug_genes(drug_file)

# === FIND INTERSECTIONS ===
print("\n=== 🔄 FINDING COMMON PROTEINS ===")

# PPI and Drug intersection
ppi_drug_common = ppi_genes.intersection(drug_genes)
print(f"\n✅ Common proteins between PPI and Drug datasets: {len(ppi_drug_common)}")
print("🔍 Sample common PPI-Drug proteins:", sorted(list(ppi_drug_common))[:10])

# PPI and Mutation intersection
ppi_mutation_common = ppi_genes.intersection(mutation_genes)
print(f"\n✅ Common proteins between PPI and Mutation datasets: {len(ppi_mutation_common)}")
print("🔍 Sample common PPI-Mutation proteins:", sorted(list(ppi_mutation_common))[:10])

# Save common proteins to files
print("\n=== 💾 SAVING RESULTS ===")

with open("ppi_drug_common_proteins.txt", "w") as f:
    for protein in sorted(ppi_drug_common):
        f.write(f"{protein}\n")
print("💾 Common PPI-Drug proteins saved to 'ppi_drug_common_proteins.txt'")

with open("ppi_mutation_common_proteins.txt", "w") as f:
    for protein in sorted(ppi_mutation_common):
        f.write(f"{protein}\n")
print("💾 Common PPI-Mutation proteins saved to 'ppi_mutation_common_proteins.txt'")

# Optional: All three datasets intersection
all_common = ppi_genes.intersection(drug_genes).intersection(mutation_genes)
print(f"\n✅ Common proteins across all three datasets: {len(all_common)}")
if all_common:
    print("🔍 Common proteins across all datasets:", sorted(list(all_common)))
    
    with open("all_common_proteins.txt", "w") as f:
        for protein in sorted(all_common):
            f.write(f"{protein}\n")
    print("💾 Common proteins across all datasets saved to 'all_common_proteins.txt'")