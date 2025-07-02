import argparse
from pathlib import Path
import imagej
import jpype
from tqdm import tqdm

def init_imagej():
    return imagej.init('sc.fiji:fiji', mode='headless')

def extract_lif_composites(ij, lif_path: Path, output_dir: Path):
    # Créer le dossier de sortie
    output_dir.mkdir(parents=True, exist_ok=True)

    # Importer les classes Java nécessaires
    BF = jpype.JClass('loci.plugins.BF')
    ImporterOptions = jpype.JClass('loci.plugins.in.ImporterOptions')

    # Options pour ouvrir toutes les séries
    opts = ImporterOptions()
    opts.setId(str(lif_path))
    opts.setOpenAllSeries(True)

    # Ouvrir toutes les séries avec Bio-Formats
    imps = BF.openImagePlus(opts)  # Liste d’objets ImagePlus Java

    base = lif_path.stem

    for i, imp in enumerate(tqdm(imps, desc=f"🖼 Séries de {lif_path.name}")):
        # Crée le nom du fichier de sortie
        output_name = f"{base}_scene{i+1}.tif"
        output_path = output_dir / output_name

        # Sauvegarde directe avec le rendu composite
        FileSaver = jpype.JClass("ij.io.FileSaver")
        fs = FileSaver(imp)
        fs.saveAsTiff(str(output_path))

def main():
    parser = argparse.ArgumentParser(description="Extraire les scènes composites RGB depuis des fichiers .lif (Leica) via PyImageJ")
    parser.add_argument("--input", "-i", type=Path, default=Path("../../data/unextracted"), help="Dossier contenant les .lif")
    parser.add_argument("--output", "-o", type=Path, default=Path("../../data/extracted/python"), help="Dossier pour enregistrer les images extraites")
    args = parser.parse_args()

    # Création du dossier de sortie
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialiser ImageJ
    ij = init_imagej()

    # Parcourir tous les fichiers .lif
    lif_files = sorted(args.input.glob("*.lif"))
    for lif_file in tqdm(lif_files, desc="📁 Fichiers .lif"):
        sub_output = args.output / lif_file.stem
        sub_output.mkdir(parents=True, exist_ok=True)
        extract_lif_composites(ij, lif_file, sub_output)

    ij.dispose()
    print("✅ Extraction terminée avec succès !")

if __name__ == "__main__":
    main()