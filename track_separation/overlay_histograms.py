import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import argparse

# Function to read .npy files and overlay histograms
def overlay_histograms(radius_values, colors,resolution):
    plt.style.use(hep.style.ATLAS)  # Use ATLAS style from mplhep
    plt.figure(figsize=(10, 6))

    for i, radius in enumerate(radius_values):
        # Load the corresponding .npy file
        distances = np.load(f"r_phi_distances_radius_{radius}.npy")
        
        fraction_below = len(distances[distances < resolution]) / len(distances)



        # Plot histogram with a different color for each radius
        plt.hist(distances, bins=50, alpha=1.0, color=colors[i],range=[0,80], label=f'r={radius} mm: {fraction_below*100:.2f} % below {resolution:.2f} mm')

    # Labeling and Legend
    plt.xlabel('xy Distance [mm]')
    plt.ylabel('Number of Tracks')
    plt.vlines(0.1, 0, 1000, colors='g', linestyles='dashed', label='0.1 mm')
    plt.title('Overlay of Track Distance Histograms at Different Radii')
    plt.legend()
    plt.savefig(f"r_phi_distance_distribution_overlay.png")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Use argparse to accept a list of radius values from the command line
    parser = argparse.ArgumentParser(description="Overlay Histograms for Different Radii")
    parser.add_argument('--radii', type=float, nargs='+', required=True, help="List of radii to overlay in mm (e.g. --radii 10 20 30)")
    parser.add_argument('--resolution',type=float,required=True,help="Assumed resolution of the detector in mm, the script will calculate the fraction of tracks below this value")

    args = parser.parse_args()

    # Define colors for different radii, you can add more colors if needed
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'gray', 'olive', 'teal', 'navy', 'lime', 'maroon']

    # Check if the number of radii matches the number of available colors
    if len(args.radii) > len(colors):
        raise ValueError("Please add more colors to the list for plotting!")

    # Call the overlay function with the given radii
    overlay_histograms(args.radii, colors, args.resolution)