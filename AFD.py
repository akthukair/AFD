import os, sys
import numpy as np
import pandas as pd
import pandas
import matplotlib.pyplot as plt
import math
from astropy.io import fits as pyfits


def plot_star(Teff, R, time, flux, n, plot_path, plot_file_name):
    xsdict = []
    count = 0

    normalized_time = time / float(time[0])
    normalized_flux = flux / float(flux[0])
    normalized_flux_list = list(normalized_flux)
    
    ## To produce figure 2a
    #plt.plot(time,normalized_flux, color='black')
    #plt.xlabel('Time (BJD-2454833)')
    #plt.ylabel('Normalized flux' r'$\;\;e^-/s$')
    #plt.title('KIC:'r'$\;$' + str(kepler_id))
    #plt.show()

    # To create a histogram of the number distribution (dc) of the flux difference between all pairs of two consecutive data points (dff) of the light curve.

    flux_difference = []
    for i in range(1, len(flux)):
        flux_difference.append(normalized_flux_list[i] - normalized_flux_list[i - 1])

    dff = []
    for i in range(n):
        dff.append(min(flux_difference) + i * (max(flux_difference) - min(flux_difference)) / (n - 1.))

    dc = [0 for _ in range(n - 1)]
    for i in range(n - 2):
        c = 0.0
        for j in range(0, len(flux) - 2):
            if flux_difference[j] < dff[i + 1] and flux_difference[j] > dff[i]:
                c = c + 1
                dc[i] = c
    
    plt.plot(dff[0:-1], dc)
    plt.xlabel('flux difference between two consecutive data points')
    plt.ylabel('Number')
    plt.show()

    # Calculate the area under the curve (A) and then determine the value of 1% of the area starting from the right (denoted by a1p)
    # Starting from the right, calculate the X-value (a1p) which encloses 0.01 fraction of A (the total area under the curve).
    # In other words, 0.99 of the total area under the curve is to the left of a1p, and 0.01 of the area is to the right.

    A = np.trapz(dc, dff[:n - 1])
    a1 = 0.99 * A
    areaB = 0
    areaA = 0
    xA = 0
    xB = 0

    area_columative = []  # Initialize a vector to hold integration results

    for i in range(n - 3):  # Loop through each data point, calculating integrals from 0 to that point

        if i == 0:
            area = 0  # Assume area under first point is zero, otherwise, calculate integral
        else:
            area = np.trapz(dc[:i], dff[:i])
            if area >= a1:
                areaA = area
                xA = dff[i - 1]
                area_columative.append(area)  # Store integral value in the cumulative vector

                break
            else:
                areaB = area
                xB = dff[i - 1]

        area_columative.append(area)  # Store integral value in the cumulative vector

    fp = [areaA, areaB]
    xp = [xA, xB]
    a1p = np.interp(a1, fp,
                    xp)  # interpolate to find a1p. Find where cumulative distribution reaches 0.99 of A this is equal to 0.01 of the area from the right.

    threshold = 3 * a1p  # threshold is equal to 3 times 1% of the area under the curve. This value is used to determine the start time of the flare.
    threshold2 = np.mean(flux_difference) + 3 * np.std(
        flux_difference)  # thrshold2 is equal to 3 standard deviations of the flux difference. This value is used to determine the end time of the flare.
    
    ## To produce figure 2c ##
    #plt.plot(dff,dc, color='black')
    #plt.plot([a1p,a1p],[0,60],color='green')
    #plt.plot([threshold,threshold],[0,60], color='crimson')
    #plt.title('KIC:'r'$\;$' + str(kepler_id))
    #plt.xlabel('Flux difference between two consecutive points')
    #plt.ylabel('Number')
    #plt.show()
    
    
    flux_avg = np.mean(normalized_flux)  # The average of normalized flux

    time_start_index = []  # To find the start time of the flare.
    flare_start_time = []
    for i in range(len(flux_difference)):
        if flux_difference[i] > threshold:
            time_start_index.append(i)
            flare_start_time.append(time[i])

    # To find the peak index , value and time of the flare.
    max_flare_index = [time_start_index[i] + np.argmax(normalized_flux[time_start_index[i]:time_start_index[i] + 5]) for
                       i in range(len(time_start_index))]
    max_flare_time = [time[i] for i in max_flare_index]
    time_flare_index = time[max_flare_index]

    time_end = []  # To find the end time of the flare.
    time_endvalue = []
    for i, element in enumerate(time_start_index):
        import scipy.interpolate as interpolate
        relative_flux = ((normalized_flux - flux_avg) / normalized_flux[time_start_index[i] - 1])  # relative flux normalized by the brightness just before the flare.
        Narr = relative_flux[max_flare_index[i] - 10:max_flare_index[i] + 30]  # part of the relative flux around the flare
        timeNarr = time[max_flare_index[i] - 10:max_flare_index[i] + 30]  # the time corresponding to Narr
        # First point is an average of five data points just before the flare, second point is an average of five data points after 5h of the peak
        # and the third point is an average of five data points 8h after the peak.
        first_point = np.average(relative_flux[time_start_index[i] - 4:time_start_index[i] + 1])
        second_point = np.average(relative_flux[max_flare_index[i] + 10:max_flare_index[i] + 15])
        third_point = np.average(relative_flux[max_flare_index[i] + 14:max_flare_index[i] + 19])

        x = np.array([time[time_start_index[i] - 2], time[max_flare_index[i] + 12], time[max_flare_index[i] + 16]])
        y = np.array([first_point, second_point, third_point])

        t, c, k = interpolate.splrep(x, y, s=0, k=2)

        N = len(Narr)
        xmin, xmax = x.min(), x.max()
        xx = np.linspace(xmin, xmax, N)

        count += 1

        spline = interpolate.BSpline(t, c, k, extrapolate=False) # Fitting a B-spline curve
        subarr = Narr - spline(xx) # subtracting the B-spline curve from the relative flux around the flare(Narr)
        
        ## To produce figure 3b ##
        #plt.plot(x, y, 'bo', label='Average points')
        #plt.plot(timeNarr, Narr)
        #plt.plot(xx, spline(xx), 'r', label='BSpline')
        #plt.scatter(timeNarr, Narr, s=10)
        #plt.legend(loc='best')
        #plt.title('KIC:' r'$\;$' + str(kepler_id))
        #plt.xlabel('Time (BJD-2454833)')
        #plt.ylabel('Relative flux'+r'${\;\;\Delta F}/{F_\circ}$')
        #plt.show()
        
        ## To produce figure 3c ##
        #plt.plot (timeNarr, subarr)
        #plt.scatter(timeNarr, subarr, s=10)
        #plt.title('KIC:' r'$\;$' + str(kepler_id))
        #plt.xlabel('Time (BJD-2454833)')
        #plt.ylabel('Relative flux' r'${\;\;\Delta F}/{F_{\circ}}$')
        #plt.show()
        
        max_flare_index2 = [np.argmax(subarr) for i in range(len(subarr))] # Determine the peak index of the subarr curve

        # Determine time end index of the flare, when subbarr after the peak become <= 3 standard deviations of the flux difference (threshold2) for the first time
        for j in range(max_flare_index2[i], max_flare_index2[i] + 24):
            if subarr[j] <= threshold2:
                time_end.append(j)
                time_endvalue.append(time[j])

                break

        xsdict.append((count, timeNarr, subarr))
    # Determine time end index of the flare on the normalized light curve: because subarr is a curve of 40 data points, the time-end-index of the flare must be converted to match the indexis of the normalized light curve
    time_end_index = [time_end[i] - max_flare_index2[i] + max_flare_index[i] for i in range(len(time_end))]
    flare_end_time = [time[i] for i in time_end_index]

    # Average of two points distributed around the flare one is the average of five data points before the start of the flare and the second is the average of five data points after the end of the flare.
    two_points_avg = []
    for i, element in enumerate(time_start_index):
        first_point_avg = np.average(normalized_flux[time_start_index[i] - 5:time_start_index[i]])
        second_point_avg = np.average(normalized_flux[time_end_index[i]:time_end_index[i] + 5])
        two_points_avg.append((first_point_avg + second_point_avg) / 2)


    #delta_flux = normalized_flux - flux_avg  # calculation of (delta_F = F-F_avg)


    # The flare amplitude = (the normalized flux at the flare peak - the normalized flux average of two points distributed around the flare)/ the normalized flux average of the lightcurve
    flare_amp = (normalized_flux[max_flare_index] - two_points_avg) / flux_avg


    ############## Flares conditions #######################
    # each line number after if statment corresponds to each flare condition:
    # 1- at lest two  data points between the peak and the end time of the flare
    # 2- If there are two consecutive data points which qualify as the start time of the flare for the same Kepler id, we choose the first point only as the beginning of the flare.
    # 3- The flare duration should be >=0.05 day
    # 4- Decline phase > Increase phase
    # 5- The flare amplitiude should be >= 0.0007
    excluded_indexes = []
    for i in range(len(time_start_index)):
        if time_end_index[i] - max_flare_index[i] >= 2 and \
           (i == 0 or time_start_index[i] - time_start_index[i - 1] > 2) and \
           (i == 0 or time_end_index[i - 1] < time_start_index[i]) and \
           flare_end_time[i] - flare_start_time[i] >= 0.05 and \
           flare_end_time[i] - max_flare_time[i] > max_flare_time[i] - flare_start_time[i] and \
           normalized_flux[time_end_index[i]] < normalized_flux[time_end_index[i] - 1] < normalized_flux[time_end_index[i] - 2] and \
           flare_amp[i] >= 0.0007:

           excluded_indexes.append(False)
        else:

           excluded_indexes.append(True)

    if False in excluded_indexes:
        x = (time[time_start_index], time[time_end_index])
        y = (normalized_flux[time_start_index], normalized_flux[time_end_index])
        plt.plot(time, normalized_flux, color='black')
        plt.plot(x, y, 'ro', color='black')
        plt.xlabel('Time (BJD-2454833)')
        plt.ylabel('Normalized flux' r'$\;\;e^-/s$')
        plt.title('KIC:'r'$\;$' + str(kepler_id))
        plt.savefig(os.path.join(plot_path, plot_file_name))
        plt.close()

        for i in range(len(xsdict)):
            plt.plot(xsdict[i][1], xsdict[i][2], color='black')
            plt.scatter(xsdict[i][1], xsdict[i][2], s=10, color='black')
            plt.title('KIC:'r'$\;$' + str(kepler_id))
            plt.xlabel('Time (BJD-2454833)')
            plt.ylabel('Relative flux' + r'${\;\;\Delta F}/{F_\circ}$')
            plt.savefig(os.path.join(plot_path, plot_file_name + '_{}.png'.format(i)))
            plt.close()

    # solving the problem of having a flare starting at the end of the file
    time_start_index = time_start_index[:len(time_end_index)]
    flare_start_time = flare_start_time[:len(flare_end_time)]
    max_flare_index = max_flare_index[:len(time_end_index)]
    max_flare_time = max_flare_time[:len(flare_end_time)]
    flare_amp = flare_amp[:len(time_end_index)]

    ########################## energy calculation in erg ###############################

    sigma = 5.6704 * 10 ** -5  # Stefan_Boltzmann constant(erg cm^-2 s^-1 k^-4)
    Tflare = 9000  # temperature of the flare =9000K
    h = 6.62606 * 10 ** -27  # plank's constant
    c = 2.99792 * 10 ** 10  # speed of light
    k = 1.38064 * 10 ** -16  # Boltzmann's constant

    KpRF = pd.read_csv('KpRF.txt')  # Kepler Instrument Response Function (high resolution)
    l = KpRF.lam * (10 ** -7)  # lambda in cm
    tr = KpRF.transmission  # Transmission

    n = len(l)

    rb1 = [] # (Kepler Response Function)*(Plank function at a given wavelength for the star)
    rb2 = [] # (Kepler Response Function)*(Plank function at a given wavelength for the flare)

    for i in range(n - 1):
        rb1.append(tr[i] * ((2.0 * h * c ** 2) / (l[i] ** 5)) * (1.0 / ((math.exp(h * c / (l[i] * k * Teff))) - 1)))
        rb2.append(tr[i] * ((2.0 * h * c ** 2) / (l[i] ** 5)) * (1.0 / ((math.exp(h * c / (l[i] * k * Tflare))) - 1)))

    s1 = np.trapz(rb1, l[:-1])
    s2 = np.trapz(rb2, l[:-1])

    Af = []  ## Area of the flare
    Lf = []  ## Luminosity of the flare

    for i in range(len(normalized_flux)):
        for i in range(len(time_start_index)):
            af = flare_amp[i] * math.pi * (R ** 2) * (s1 / s2)
            Af.append(flare_amp[i] * math.pi * (R ** 2) * (s1 / s2))
            Lf.append(sigma * (Tflare ** 4) * af)

    energy = [] # The total energy of the flare is the integral of (Lf) over the flare duration

    for i in range(len(time_start_index)):
        flare_energy = np.trapz(np.array(Lf[time_start_index[i]:time_end_index[i] + 1]),time[time_start_index[i]: time_end_index[i] + 1])
        flare_energy_per_second = flare_energy * 24 * 60 * 60
        energy.append(flare_energy_per_second)

    return time_start_index, time_end_index, flare_start_time, flare_end_time, energy, excluded_indexes, max_flare_index, max_flare_time, flare_amp

    ###################################

path = 'C:\\Users\\.......\\fits files folder name' # Users need to modify the fits files folder path
plots_path = 'C:\\Users\\.......\\plots folder name'# Users need to modify the plots folder path
file_list = os.listdir(path)
kepler_id_list = []
quarter_list = []
Teff_list = []
Radius_list = []
logg_list = []
Prot_list = []
time_start_index_list = []
time_end_index_list = []
flare_start_time_list = []
flare_end_time_list = []
energy_list = []
excluded_indexes_list = []
max_flare_index_list = []
max_flare_time_list = []
flare_amp_list = []

s_p = pd.read_csv('G-type.csv') #  stellar parameters for A,F,G,K,M types stars taken from q1_q17_dr25_stellar catalog downloaded from https://exoplanetarchive.ipac.caltech.edu
#########################
import traceback

errors = []


for file_name in file_list:
    print file_name
    with open(os.path.join(path, file_name)) as fits_file:
        try:
            pyfits_object = pyfits.open(fits_file)
        except:
            print 'pyfits error when openning ' + file_name
            import traceback

            print traceback.format_exc()
            continue

        time = pyfits_object[1].data['TIME']
        flux = pyfits_object[1].data['PDCSAP_FLUX']
        kepler_id = pyfits_object[1].header['KEPLERID']
        sp_series = s_p[s_p['kepler_id'] == kepler_id]

        if len(sp_series) > 0:
            radius = sp_series.Radius.iloc[0]
            Teff = sp_series.Teff.iloc[0]
            logg = sp_series.logg.iloc[0]
            Prot = sp_series.Prot.iloc[0]
        else:
            Teff = pyfits_object[0].header['TEFF']  # temperature of the star(k)
            radius = pyfits_object[0].header['RADIUS']
            logg = pyfits_object[0].header['LOGG']
            Prot = ''
        quarter = pyfits_object[0].header['QUARTER']
        R = 6.957 * radius * 10 ** 10  # star's radius(cm)
        n = 1000

    dataframe = pandas.DataFrame(zip(time, flux)).dropna()
    time = dataframe[0].values
    flux = dataframe[1].values
    print kepler_id

    ###################################################
    plot_file_name = file_name.replace('.fits', '.png')
    ##########################################
    try:
        time_start_index, time_end_index, flare_start_time, flare_end_time, energy, excluded_indexes, max_flare_index, max_flare_time, flare_amp = plot_star(
            Teff, R, time, flux, n, plots_path, plot_file_name)
    except:
        errors.append({'file_name': file_name, 'traceback': traceback.format_exc()})
        continue
    ########################################################
    kepler_id_list.extend([kepler_id for _ in range(len(time_start_index))])
    quarter_list.extend([quarter for _ in range(len(time_start_index))])
    Teff_list.extend([Teff for _ in range(len(time_start_index))])
    Radius_list.extend([radius for _ in range(len(time_start_index))])
    logg_list.extend([logg for _ in range(len(time_start_index))])
    Prot_list.extend([Prot for _ in range(len(time_start_index))])
    time_start_index_list.extend(time_start_index)
    time_end_index_list.extend(time_end_index)
    flare_start_time_list.extend(flare_start_time)
    flare_end_time_list.extend(flare_end_time)
    energy_list.extend(energy)
    excluded_indexes_list.extend(excluded_indexes)
    max_flare_time_list.extend(max_flare_time)
    max_flare_index_list.extend(max_flare_index)
    flare_amp_list.extend(flare_amp)
    print '---------------------------------------'
final_data_frame = pandas.DataFrame(
    zip(kepler_id_list, quarter_list, time_start_index_list, time_end_index_list, flare_start_time_list,
        flare_end_time_list, max_flare_index_list, max_flare_time_list, energy_list, excluded_indexes_list,
        Teff_list,
        Radius_list, logg_list, Prot_list, flare_amp_list),
    columns=['kepler_id', 'quarter', 'time_start_index', 'time_end_index', 'flare_start_time', 'flare_end_time',
                 'flare_peak_index', 'flare_peak_time', 'energy', 'excluded', 'Teff', 'Radius', 'logg', 'Prot',
                 'flare_amp'])

final_data_frame['flare_duration'] = final_data_frame.flare_end_time - final_data_frame.flare_start_time
final_data_frame['time_start_BJD'] = final_data_frame.flare_start_time + 54833
final_data_frame['time_end_BJD'] = final_data_frame.flare_end_time + 54833
final_data_frame['flare_peak_BJD'] = final_data_frame.flare_peak_time + 54833

final_data_frame.to_csv('flares-candidates.csv') # CSV file showing all flare candidates captured by the code when the flux difference exceeded the threshold limit

filtered_df = final_data_frame[(final_data_frame.excluded == False)]
filtered_df.to_csv('final-flares.csv') # CSV file showing only flares that met all the conditions
#############################################
errors_df = pd.DataFrame(errors)
errors_df.to_csv( 'errors.csv') # CSV file showing errors