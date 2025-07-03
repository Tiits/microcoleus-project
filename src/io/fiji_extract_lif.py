#@ File (label="Source folder (.lif)", style="directory") srcDir
#@ File (label="Destination folder", style="directory") dstDir

from ij import IJ, ImagePlus
from loci.plugins import BF
from loci.plugins.in import ImporterOptions
import os

def make_options(path):
    opts = ImporterOptions()
    opts.setId(path)
    opts.setOpenAllSeries(True)       # Ouvrir toutes les séries du fichier
    opts.setAutoscale(True)           # Appliquer un contraste automatique
    opts.setSplitChannels(False)      # Ne pas séparer les canaux
    opts.setSplitTimepoints(False)    # Ne pas séparer les points temporels
    opts.setSplitFocalPlanes(False)   # Ne pas séparer les plans focaux
    return opts

# Parcours de tous les fichiers .lif dans le dossier source
for filename in os.listdir(str(srcDir)):
    if not filename.lower().endswith(".lif"):
        continue
    base_name = os.path.splitext(filename)[0]
    input_path = os.path.join(str(srcDir), filename)
    output_folder = os.path.join(str(dstDir), base_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Chargement de l’image via Bio-Formats
    options = make_options(input_path)
    image_list = BF.openImagePlus(options)

    # Traitement de chaque série importée
    for idx, img in enumerate(image_list, start=1):
        if img.getNChannels() == 1:
            # Dupliquer pour créer un hyperstack RGB à partir d’un seul canal
            duplicate = img.duplicate()
            IJ.run(duplicate, "Stack to Hyperstack...", "order=xyzct channels=3 slices=1 frames=1 display=Color")
            IJ.run(duplicate, "RGB Color", "")
            rgb_image = duplicate
        else:
            # Convertir un hyperstack multi-canaux en composite puis en RGB
            IJ.run(img, "Make Composite", "")
            IJ.run(img, "RGB Color", "")
            rgb_image = img

        # Construction du nom de fichier: [nom du fichier .lif]_scene[x]_fiji.tif
        scene_name = "%s_scene%d_fiji.tif" % (base_name, idx)
        output_path = os.path.join(output_folder, scene_name)
        IJ.saveAs(rgb_image, "Tiff", output_path)

        # Fermeture sans invite de sauvegarde
        rgb_image.changes = False
        rgb_image.close()

# Message de confirmation à l’utilisateur
IJ.showMessage("Done", "All LIF files have been exported as color TIFFs!")
