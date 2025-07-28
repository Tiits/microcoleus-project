"""
Extract LIF Composites Module

This module provides functions to initialize PyImageJ, extract composite RGB scenes from Leica .lif files, and save them as TIFF images.
"""
import argparse
from pathlib import Path

import imagej
import jpype
from tqdm import tqdm


def init_imagej():
    """
    Initialize and return a headless ImageJ gateway for image processing.

    Returns:
        imagej.ImageJ: Initialized ImageJ instance in headless mode.
    """
    return imagej.init('sc.fiji:fiji', mode='headless')


def extract_lif_composites(ij, lif_path: Path, output_dir: Path):
    """
    Extract composite scenes from a Leica .lif file and save each as a TIFF image.

    Parameters:
        ij (ImageJ): Initialized PyImageJ instance.
        lif_path (Path): Path to the input .lif file.
        output_dir (Path): Directory where extracted TIFF images will be saved.
    """
    # Ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import Bio-Formats classes via JPype.
    BF = jpype.JClass('loci.plugins.BF')
    ImporterOptions = jpype.JClass('loci.plugins.in.ImporterOptions')

    # Configure ImporterOptions to read all image series from the LIF file.
    opts = ImporterOptions()
    opts.setId(str(lif_path))
    opts.setOpenAllSeries(True)

    # Open all series as ImagePlus objects.
    imps = BF.openImagePlus(opts)

    # Determine base filename for output naming.
    base = lif_path.stem

    for i, imp in enumerate(tqdm(imps, desc=f"🖼 Séries de {lif_path.name}")):
        # Generate output filename with scene index.
        output_name = f"{base}_scene{i + 1}.tif"
        output_path = output_dir / output_name

        # Initialize ImageJ FileSaver to write TIFF files.
        FileSaver = jpype.JClass("ij.io.FileSaver")
        fs = FileSaver(imp)
        # Save the current scene as a TIFF image.
        fs.saveAsTiff(str(output_path))


def main():
    """
    Parse command-line arguments, initialize ImageJ, and extract scenes from .lif files.

    Steps:
    1. Parse input and output directory arguments.
    2. Ensure output directories exist.
    3. Initialize ImageJ in headless mode.
    4. Iterate through .lif files to extract scenes.
    5. Dispose of the ImageJ instance and print completion.
    """
    # Create argument parser for CLI options.
    parser = argparse.ArgumentParser(
        description="Extraire les scènes composites RGB depuis des fichiers .lif (Leica) via PyImageJ")
    # Define input and output directory arguments.
    parser.add_argument("--input", "-i", type=Path, default=Path("../../data/unextracted"),
                        help="Dossier contenant les .lif")
    parser.add_argument("--output", "-o", type=Path, default=Path("../../data/extracted/python"),
                        help="Dossier pour enregistrer les images extraites")
    # Parse the provided arguments.
    args = parser.parse_args()

    # Create base output directory if it doesn't exist.
    args.output.mkdir(parents=True, exist_ok=True)

    # Initialize the ImageJ gateway.
    ij = init_imagej()

    # Gather all .lif files from the input directory.
    lif_files = sorted(args.input.glob("*.lif"))
    for lif_file in tqdm(lif_files, desc="📁 Fichiers .lif"):
        # Create a subdirectory for each LIF file's extracted scenes.
        sub_output = args.output / lif_file.stem
        sub_output.mkdir(parents=True, exist_ok=True)
        # Extract and save composite scenes for this LIF file.
        extract_lif_composites(ij, lif_file, sub_output)

    # Dispose of the ImageJ instance to free resources.
    ij.dispose()
    # Notify user of successful extraction.
    print("✅ Extraction terminée avec succès !")


if __name__ == "__main__":
    main()
