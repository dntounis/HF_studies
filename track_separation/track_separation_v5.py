import numpy as np
import uproot
import mplhep as hep
import matplotlib.pyplot as plt
import awkward as ak
import argparse 


#Jim: to translate from trackstate parametrization to helix parametrization use information see:
#https://github.com/key4hep/k4Pandora/blob/583288153bf6e6f1ea94a0d5b1bcb515628b7a4d/Utility/MarlinUtil/01-08/source/HelixClass.cc#L91

#Jim: track params in edm4hep, inherited from LCIO, using perigee parametrization:
#https://bib-pubdb1.desy.de/record/81214/files/LC-DET-2006-004%5B1%5D.pdf

FCT = 2.99792458E-4  # Constant used in momentum calculation


# Function to calculate transverse momentum (pT) from omega
def calculate_track_pt(omega):
    return 1 / np.abs(omega)

# Function to project track's x, y position at given radius r
def helix_track_position_momentum(D0, phi0, omega, Z0, tanLambda, radius_to_project,B):

    """
    Compute the x, y, z positions of a track at a given radius
    
    Parameters:
    - D0: transverse impact parameter [mm]
    - Z0: longitudinal impact parameter [mm]
    - phi0: azimuthal angle at point of closest approach [rad]
    - omega: signed curvature of the track [1/mm]
    - tanLambda: tangent of the dip angle of the track in r-z plane
    - B: magnetic field [T]
    - radius_to_project: target radius in the transverse plane [mm]
    
    Returns:
    - x, y, z: positions at the projected radius in mm
    """



    # Calculate radius of curvature in transverse plane
    charge = np.sign(omega)
    radius = 1.0 / np.abs(omega)  # [mm] radius of curvature in the transverse plane

    # Transverse impact parameter (PCA - point of closest approach) coordinates
    x_PCA = -D0 * np.sin(phi0)
    y_PCA = D0 * np.cos(phi0)

    # Transverse momentum (pT) calculation
    p_xy = FCT * B * radius  # Transverse momentum in the xy-plane
    
    # Momentum components
    p_x = p_xy * np.cos(phi0)
    p_y = p_xy * np.sin(phi0)
    p_z = tanLambda * p_xy

    # Center of the helix in the transverse plane
    x_center = x_PCA + radius * np.cos(phi0 - np.pi / 2 * charge)
    y_center = y_PCA + radius * np.sin(phi0 - np.pi / 2 * charge)

    # Distance from the center to the IP (interaction point)
    dist_center_to_IP = np.sqrt(x_center**2 + y_center**2)

    # Solve for the points where the helix intersects the projection circle
    phi_center = np.arctan2(y_center, x_center)
    phi_star = (radius_to_project**2 + dist_center_to_IP**2 - radius**2) / (2 * radius_to_project * dist_center_to_IP)
    
    # Handle edge cases where the projection doesn't intersect
    phi_star = np.clip(phi_star, -1.0, 1.0)  # Ensure within [-1, 1] for acos
    
    # Calculate the angle for the intersection points
    delta_phi = np.arccos(phi_star)
    
    # Calculate the two possible intersection points
    x1 = radius_to_project * np.cos(phi_center + delta_phi)
    y1 = radius_to_project * np.sin(phi_center + delta_phi)
    
    x2 = radius_to_project * np.cos(phi_center - delta_phi)
    y2 = radius_to_project * np.sin(phi_center - delta_phi)
    
    # Calculate the corresponding phi values
    phi1 = np.arctan2(y1 - y_center, x1 - x_center)
    phi2 = np.arctan2(y2 - y_center, x2 - x_center)
    
    phi0_ref = np.arctan2(-y_center, -x_center)
    
    dphi1 = phi1 - phi0_ref
    dphi2 = phi2 - phi0_ref

    # Adjust angles based on charge sign
    if charge < 0:
        if dphi1 < 0:
            dphi1 += 2 * np.pi
        if dphi2 < 0:
            dphi2 += 2 * np.pi
    else:
        if dphi1 > 0:
            dphi1 -= 2 * np.pi
        if dphi2 > 0:
            dphi2 -= 2 * np.pi

    # Calculate the "times" associated with the two points
    time1 = -charge * dphi1 * radius / p_xy
    time2 = -charge * dphi2 * radius / p_xy

    # Choose the correct intersection point based on the "time" (take the shorter one)
    if time1 < time2:
        x = x1
        y = y1
        time = time1
    else:
        x = x2
        y = y2
        time = time2

    # Calculate the z position based on time and tanLambda
    z = Z0 + time * p_xy * tanLambda

        
    return x, y, z, p_x, p_y, p_z

def project_helix_to_radius(D0, phi, omega, r):
    xc = (1 / omega) * np.sin(phi)
    yc = -(1 / omega) * np.cos(phi)

    # Project x and y at radius r
    x = xc + (r + D0) * np.cos(phi)
    y = yc + (r + D0) * np.sin(phi)

    return x, y

# Function to calculate r-phi distance between two tracks
def r_phi_distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# Function to compute eta from px, py, pz
def compute_eta(px, py, pz):
    #p = np.sqrt(px**2 + py**2 + pz**2)
    theta = np.arctan2(np.sqrt(px**2 + py**2), pz)
    eta = -np.log(np.tan(theta / 2.0))
    return eta

# Function to compute phi from px, py
def compute_phi(px, py):
    phi = np.arctan2(py, px)
    return phi

# Function to compute delta R
def delta_r(eta1, phi1, eta2, phi2):
    delta_eta = eta1 - eta2
    delta_phi = np.abs(phi1 - phi2)
    if delta_phi > np.pi:
        delta_phi = 2*np.pi - delta_phi
    return np.sqrt(delta_eta**2 + delta_phi**2)

#def calculate_momentum(pt, tanLambda):
#    pz = pt * tanLambda  # Calculate longitudinal momentum
#    p = np.sqrt(pt**2 + pz**2)  # Calculate total momentum (magnitude of the 3D momentum vector)
#    return p, pz

# Function to calculate transverse momentum (pT) and longitudinal momentum (pz)
def calculate_momentum(omega, tanLambda, B=2.0):
    # pT = B / |omega| assuming magnetic field B is 2 Tesla (you can adjust this)
    pT = B / np.abs(omega)
    
    # pz is related to pT and tanLambda
    pz = pT * tanLambda
    
    return pT, pz

# Main function to process the events
def process_events(radius_to_project):

    # Load the event file using uproot
    file = uproot.open("/fs/ddn/sdf/group/atlas/d/dntounis/Hss_setup_test/Delphes/zhiggs_bb/zhiggs_nunu_bb_DELPHES_IDEA_Winter2023_9.edm4hep.root")  
    events = file["events"]

    # Count the number of events
    #n_events = len(events["Jet.momentum.x"].array())
    n_events = 5000


    print("n events = ", n_events)

    #print(events.keys())  # List all branches available in the events object

    jet_px = events["Jet.momentum.x"].array()
    jet_py = events["Jet.momentum.y"].array()
    jet_pz = events["Jet.momentum.z"].array()
    jet_energy = events["Jet.energy"].array()

    track_d0 = events["EFlowTrack_1.D0"].array()
    track_z0 = events["EFlowTrack_1.Z0"].array()
    track_omega = events["EFlowTrack_1.omega"].array()
    track_phi = events["EFlowTrack_1.phi"].array()
    track_tan_lambda = events["EFlowTrack_1.tanLambda"].array()

    # Use ak.num to get the number of jets per event
    jet_px_counts = ak.num(jet_px)  # This will return the number of jets in each event

    print("Total number of jets in all events: ", len(ak.flatten(jet_px)))

    print("Number of jets in each event:", jet_px_counts)


    # Radii for projection
    #radius_to_project = 10 # in mm!!! Different values of r to explore

    # Prepare a dictionary to store r-phi distances for each radius
    r_phi_dist_at_radius_array = []

    dijet_mass_array = []


    # Main loop over events
    for event in range(n_events):

        #jets_in_event = jet_px_counts[event]
        jets_in_event = len(jet_px[event])
        tracks_in_event = len(track_omega[event])

        if jets_in_event < 2:
            print(f"Skipping event {event} with {jets_in_event} jets")
            continue

        dijet_inv_mass = np.sqrt((jet_energy[event][0] + jet_energy[event][1])**2 - (jet_px[event][0] + jet_px[event][1])**2 - (jet_py[event][0] + jet_py[event][1])**2 - (jet_pz[event][0] + jet_pz[event][1])**2)
        dijet_mass_array.append(dijet_inv_mass)
    
        if event%100 == 0:
            print("------- Event  ",event,"-------")
            print("Number of tracks in event = ",tracks_in_event)
            print(f"Event {event}: dijet invariant mass = {dijet_inv_mass} GeV")


        for jet_idx in range(2):  # Each event has two jets
            # Calculate jet pT, eta, and phi
            jet_pt = np.sqrt(jet_px[event][jet_idx]**2 + jet_py[event][jet_idx]**2)
            jet_eta = compute_eta(jet_px[event][jet_idx], jet_py[event][jet_idx], jet_pz[event][jet_idx])
            jet_phi = compute_phi(jet_px[event][jet_idx], jet_py[event][jet_idx])

            if event%1000 == 0:
                print(f"Jet {jet_idx} in Event {event}: pT = {jet_pt}, eta = {jet_eta}, phi = {jet_phi}")

            # Initialize variables to store the highest pT track info
            highest_p = -np.inf
            highest_momentum_idx = -1
            highest_track_coords = None
            track_coords = []  # To store (x, y) positions of all tracks


            for track_idx in range(tracks_in_event):

                tr_d0 = track_d0[event][track_idx]
                tr_z0 = track_z0[event][track_idx]
                tr_phi = track_phi[event][track_idx]
                tr_omega = track_omega[event][track_idx]
                tr_tanLambda = track_tan_lambda[event][track_idx]

                tr_x,tr_y,tr_z,tr_px,tr_py,tr_pz = helix_track_position_momentum(tr_d0, tr_phi, tr_omega, tr_z0, tr_tanLambda, radius_to_project, 2.0)
                #tr_pt, tr_pz = calculate_momentum(tr_omega, tr_tanLambda)
                tr_pt = np.sqrt(tr_px**2 + tr_py**2)
                tr_p = np.sqrt(tr_px**2 + tr_py**2 + tr_pz**2)
                # Calculate transverse momentum, full momentum, and longitudinal momentum
                #tr_pt = calculate_track_pt(tr_omega)
                #tr_p, tr_pz = calculate_momentum(tr_pt, tr_tanLambda)

                #tr_px = tr_pt*np.cos(tr_phi)
                #tr_py = tr_pt*np.sin(tr_phi)
                tr_eta = compute_eta(tr_px,tr_py,tr_pz)
                dr = delta_r(jet_eta,jet_phi,tr_eta,tr_phi)
                #print(f"Track {track_idx} in Event {event}: pT = {tr_pt:.3f}, p = {tr_p:.3f}, pz = {tr_pz:.3f}, phi = {tr_phi:.3f}, eta = {tr_eta:.3f}")
                #if dr <= 1.0:
                if dr <= 1.0 and tr_pt > 0.1 and np.abs(tr_eta)<2.55 and np.abs(tr_z)<2000: # keep tracks in the tracker region!!
                    if event%1000 == 0:
                        print(f"Track {track_idx} in Event {event}:pT = {tr_pt:.3f}, p = {tr_p:.3f}, pz = {tr_pz:.3f}, phi = {tr_phi:.3f}, eta = {tr_eta:.3f}, x = {tr_x:.3f} mm, y = {tr_y:.3f} mm, z = {tr_z:.3f}, D0 = {tr_d0:.3f} mm, Z0 = {tr_z0:.3f} mm, omega = {tr_d0:.3f} mm-1, tanLambda = {tr_tanLambda:.3f}  is within ΔR = 0.4 of Jet {jet_idx}: ΔR = {dr}")

                    track_coords.append([tr_x,tr_y])

                    if tr_p > highest_p:
                        highest_p = tr_p
                        highest_momentum_idx = track_idx
                        highest_track_coords = tr_x,tr_y
                


            for track_coord in track_coords:
                r_phi_dist = r_phi_distance(highest_track_coords[0], highest_track_coords[1], track_coord[0], track_coord[1])
                if r_phi_dist != 0: # Avoid self-comparison
                    r_phi_dist_at_radius_array.append(r_phi_dist)

    return dijet_mass_array,r_phi_dist_at_radius_array



# Use argparse to allow passing arguments from the command line
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track Separation Script")
    parser.add_argument('--radius', type=float, required=True, help="Radius to project in mm")
    
    args = parser.parse_args()

    dijet_masses, r_phi_distances = process_events(args.radius)

    # plot dijet invariant mass
    plt.figure(figsize=(10, 6))
    plt.hist(dijet_masses, bins=50, alpha=0.6, label=f'Dijet Invariant Mass')
    plt.xlabel('Dijet Invariant Mass [GeV]')
    plt.ylabel('Number of Events')
    plt.title('Dijet Invariant Mass Distribution')
    plt.savefig("dijet_mass_distribution_v3.png")
    plt.show()


    # Plot the histogram
    plt.style.use(hep.style.ATLAS)  # Use ATLAS style from mplhep
    plt.figure(figsize=(10, 6))
    plt.hist(r_phi_distances, bins=50, alpha=0.6, label=f'xy Distance at r={args.radius} mm')
    plt.xlabel('xy Distance [mm]')
    plt.ylabel('Number of Tracks')
    plt.title(f'Distance between Highest Momentum Track and Others at r={args.radius} mm')
    plt.savefig(f"r_phi_distance_distribution_radius_{args.radius}.png")
    plt.legend()
    plt.show()

    # Optionally, you could save this data to a file for external use
    np.save(f"r_phi_distances_radius_{args.radius}.npy", r_phi_distances)
