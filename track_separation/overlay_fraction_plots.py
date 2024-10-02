import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import argparse

# Function to read .npy files and overlay histograms
def overlay_fraction_plots(radius_values, colors):
    plt.style.use(hep.style.ATLAS)  # Use ATLAS style from mplhep
    plt.figure(figsize=(10, 6))

    resolution_array = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] # in mm

    for i, radius in enumerate(radius_values):
        # Load the corresponding .npy file
        distances = np.load(f"r_phi_distances_radius_{radius}.npy")
        
        array_fraction_below = []
        for res in resolution_array:
            array_fraction_below.append(len(distances[distances < res]) / len(distances))
            array_percent_below = [x*100 for x in array_fraction_below]

        # Plot histogram with a different color for each radius
        #plt.plot(resolution_array,array_fraction_below,marker='o',color=colors[i],label=f'r={radius} mm')
        plt.plot(resolution_array,array_percent_below,marker='o',color=colors[i],label=f'radial distance={radius} mm from beamline')

    # Labeling and Legend
    plt.xlabel('Track xy Separation [mm]')
    #$plt.ylabel('Fraction of Tracks')
    #plt.title('Fraction of tracks with separation below a given distance')
    plt.vlines(10, 0, 70, colors='g', linestyles='dashed', label='1 cm')
    plt.ylabel('Percentage [%]')
    plt.title('% of tracks with  xy separation below a given distance')    
    plt.legend()
    plt.ylim(-1,70)
    plt.savefig(f"r_phi_distance_fraction_overlay.png")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Use argparse to accept a list of radius values from the command line
    parser = argparse.ArgumentParser(description="Overlay Histograms for Different Radii")
    parser.add_argument('--radii', type=float, nargs='+', required=True, help="List of radii to overlay in mm (e.g. --radii 10 20 30)")

    args = parser.parse_args()

    # Define colors for different radii, you can add more colors if needed
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray', 'olive', 'teal', 'navy', 'lime', 'maroon']

    # Check if the number of radii matches the number of available colors
    if len(args.radii) > len(colors):
        raise ValueError("Please add more colors to the list for plotting!")

    # Call the overlay function with the given radii
    overlay_fraction_plots(args.radii, colors)