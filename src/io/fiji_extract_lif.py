# Script: fiji_extract_lif.py
# Description: Extract composite RGB scenes from Leica .lif files using Bio-Formats in Fiji and save as TIFF images.
# Usage: Run this script in the Fiji Script Editor with the source (`srcDir`) and destination (`dstDir`) directories specified.

# @ File (label="Source folder (.lif)", style="directory") srcDir
# @ File (label="Destination folder", style="directory") dstDir

from loci.plugins.
from ij import IJ, ImagePlus
from loci.plugins import BF
in import ImporterOptions
import os


# Function to configure Bio-Formats ImporterOptions for a given .lif file path.
def make_options(path):
    # Set the path to the input .lif file.
    opts = ImporterOptions()
    opts.setId(path)
    # Instruct Bio-Formats to load all image series in the file.
    opts.setOpenAllSeries(True)
    # Enable intensity autoscaling for display.
    opts.setAutoscale(True)
    # Do not split separate channels into different images.
    opts.setSplitChannels(False)
    # Do not split timepoints into separate images.
    opts.setSplitTimepoints(False)
    # Do not split focal planes into separate images.
    opts.setSplitFocalPlanes(False)
    return opts


# Iterate over each file in the source directory.
for filename in os.listdir(str(srcDir)):
    # Skip files that do not have a .lif extension.
    if not filename.lower().endswith(".lif"):
        continue
    # Determine the base filename without extension for naming outputs.
    base_name = os.path.splitext(filename)[0]
    # Construct the full path to the input .lif file.
    input_path = os.path.join(str(srcDir), filename)
    # Define the directory where extracted scenes will be saved.
    output_folder = os.path.join(str(dstDir), base_name)
    # Create the output folder if it does not already exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize Bio-Formats importer options for this file.
    options = make_options(input_path)
    # Open the .lif file and load all series as ImagePlus objects.
    image_list = BF.openImagePlus(options)

    # Loop through each image series (scene) in the LIF file.
    for idx, img in enumerate(image_list, start=1):
        # Check if the scene is single-channel (grayscale).
        if img.getNChannels() == 1:
            # Duplicate the grayscale image for composite conversion.
            duplicate = img.duplicate()
            # Convert the stack to a 3-channel RGB hyperstack and apply RGB color lookup.
            IJ.run(duplicate, "Stack to Hyperstack...", "order=xyzct channels=3 slices=1 frames=1 display=Color")
            IJ.run(duplicate, "RGB Color", "")
            rgb_image = duplicate
        # For multi-channel images, make a composite and apply RGB coloring.
        else:
            IJ.run(img, "Make Composite", "")
            IJ.run(img, "RGB Color", "")
            rgb_image = img

        # Save the RGB image as a TIFF file to the output path.
        scene_name = "%s_scene%d_fiji.tif" % (base_name, idx)
        output_path = os.path.join(output_folder, scene_name)
        IJ.saveAs(rgb_image, "Tiff", output_path)

        # Mark image as unchanged to avoid save prompts.
        rgb_image.changes = False
        # Close the image to release memory.
        rgb_image.close()

# Display a message to the user indicating completion of the export process.
IJ.showMessage("Done", "All LIF files have been exported as color TIFFs!")
