from pathlib import Path
import numpy as np
import uvnav_py 




# run = "skip2_id-init"
# folder = "/home/arturo/Documents/voyis_tools/feature_matching/results/nonplanar_traj"



skips = [5, 10, 15]

icp_skips = []
dir_skips = []
fou_skips = []
rad_skips = []
teas_skips = []
for skip in skips:
    folder = "/media/arturo/Voyis/OceanSIM/welland_boat_planar/reg_skip"+ str(skip)
    # folder = "/media/arturo/Voyis/OceanSIM/planar_example"

    gamma = 1088 / (
                        2.0 * 3.0 * np.sin(np.deg2rad(130) / 2.0)
                    )
    """
    Load results from the specified run and folder.
    """
    icp_results = np.load(f"{folder}/icp_results.npz")

    gt_results = np.load(f"{folder}/gt_results.npz")
    dir_results = np.load(f"{folder}/direct_results.npz")
    fou_results = np.load(f"{folder}/fourier_results.npz")
    try:
        rad_results = np.load(f"{folder}/radon_results.npz")

    except FileNotFoundError:
        print(f"Radon results not found in {folder}. Skipping Radon results.")
        rad_results = None

    try:
        teaser_results = np.load(f"{folder}/teaser_results.npz")

    except FileNotFoundError:
        print(f"TEASER results not found in {folder}. Skipping TEASER results.")
        teaser_results = None

    icp_se3 = icp_results['icp_se3']
    gt_se3 = gt_results['gt_se3']
    dir_se3 = dir_results['dir_se3']
    fou_se3 = fou_results['fou_se3'] 
    fou_se3[:, 3:] /= gamma  # Scale the shift by gamma
    rad_se3 = rad_results['rad_se3']
    rad_se3[:, 3:] /= gamma  # Scale the shift by gamma
    teas_se3 = teaser_results['teas_se3']
    teas_se3[:, 3:] /= gamma  # Scale the shift by gamma

    try:
        init_results = np.load(f"{folder}/init_results.npz")
        init_se3 = init_results['init_se3']
    except FileNotFoundError:
        print(f"Init results not found in {folder} Skipping init results.")
        init_se3 = np.zeros(gt_se3.shape, dtype=np.float32)



    print(f"Total number of nans in icp_se3: {np.isnan(icp_se3[:, 0]).sum()}")

    print(f"nans in direct_se3: {np.isnan(dir_se3[:, 0]).sum()}")




    error_icp = icp_se3 - gt_se3
    error_dir = dir_se3 - gt_se3
    error_fou = fou_se3 - gt_se3
    error_rad = rad_se3 - gt_se3
    error_teas = teas_se3 - gt_se3
    error_init = init_se3-gt_se3  # Assuming gt_se3 is the ground truth and we want to see the error from it

    # Exclude NaN values from the error calculations
    error_icp_m = error_icp[~np.isnan(error_icp).any(axis=1)]
    error_dir_m = error_dir[~np.isnan(error_dir).any(axis=1)]
    error_fou_m = error_fou[~np.isnan(error_fou).any(axis=1)]
    error_rad_m = error_rad[~np.isnan(error_rad).any(axis=1)]
    error_teas_m = error_teas[~np.isnan(error_teas).any(axis=1)]
    error_init_m = error_init[~np.isnan(error_init).any(axis=1)]

    print(f"Mean ICP error", np.mean(error_icp_m, axis=0))
    print(f"Mean Direct error", np.mean(error_dir_m, axis=0))
    print(f"Mean Fourier error", np.mean(error_fou_m, axis=0))
    print(f"Mean Radon error", np.mean(error_rad_m, axis=0))
    print(f"Mean TEASER error", np.mean(error_teas_m, axis=0))
    print(f"Mean Init error", np.mean(error_init_m, axis=0))

    print(f"Std ICP error", np.std(error_icp_m, axis=0))
    print(f"Std Direct error", np.std(error_dir_m, axis=0))
    print(f"Std Fourier error", np.std(error_fou_m, axis=0))
    print(f"Std Radon error", np.std(error_rad_m, axis=0))
    print(f"Std TEASER error", np.std(error_teas_m, axis=0))
    print(f"Std Init error", np.std(error_init_m, axis=0))


    print(f"Average motion between frames (rad, m):", np.rad2deg(np.mean(np.abs(gt_se3[:,2]))), np.mean(np.linalg.norm(gt_se3[:, 3:5], axis=1)))
    error_sq_icp = np.sqrt(error_icp_m**2)
    error_sq_fou = np.sqrt(error_fou_m**2)
    error_sq_dir = np.sqrt(error_dir_m**2)
    error_sq_rad = np.sqrt(error_rad_m**2)
    error_sq_init = np.sqrt(error_init_m**2)
    error_sq_teas = np.sqrt(error_teas_m**2)


    icp_skips.append(error_sq_icp)
    dir_skips.append(error_sq_dir)
    fou_skips.append(error_sq_fou)
    rad_skips.append(error_sq_rad)
    teas_skips.append(error_sq_teas)
    

    ## Make a latex table of the mean and std errors
    # print("\n\\begin{table}[h]")
    # print("\\centering")
    # print("\\begin{tabular}{|c|c|c|c|}")
    # print("\\hline")
    # print("Method & Mean Error & Std Error \\\\") 
    # print("\\hline")
    # methods = ['ICP', 'Fourier', 'Init']
    # for i, method in enumerate(methods):    
    #     mean_error = np.mean([error_icp_m, error_fou_m, error_init_m][i], axis=0)
    #     std_error = np.std([error_icp_m, error_fou_m, error_init_m][i], axis=0)
    #     print(f"{method} & {mean_error[0]:.3f} & {std_error[0]:.3f} \\\\")
    #     print(f"{method} & {mean_error[1]:.3f} & {std_error[1]:.3f} \\\\")
    #     print(f"{method} & {mean_error[2]:.3f} & {std_error[2]:.3f} \\\\")
    # print("\\hline")
    # print("\\end{tabular}")
    # print("\\caption{Mean and standard deviation of registration errors for different methods.}")
    # print("\\label{tab:registration_errors}")
    # print("\\end{table}\n")


# # Plot a scatter plot of the errors
import matplotlib.pyplot as plt

# plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 14})
plt.rcParams.update({'text.usetex': True})

# fig, ax = plt.subplots(3, 1, figsize=(12, 6))

# for i in range(3):
#     ax[i].scatter(range(len(error_icp)), error_icp[:, 2+i], label='ICP', alpha=0.5, color='blue')
#     ax[i].scatter(range(len(error_fou)), error_fou[:, 2+i], label='Fourier', alpha=0.5, color='orange')
#     ax[i].scatter(range(len(error_dir)), error_dir[:, 2+i], label='Direct', alpha=0.5, color='red')
#     ax[i].scatter(range(len(error_rad)), error_rad[:, 2+i], label='Radon', alpha=0.5, color='purple')
#     ax[i].scatter(range(len(error_teas)), error_teas[:, 2+i], label='TEASER', alpha=0.5, color='brown')
#     # ax[i].scatter(range(len(error_init)), error_init[:, 2+i], label='Init', alpha=0.5, color='green')
#     # ax[i].set_xlabel('Timestep')
    
    
#     ax[i].legend()
#     ax[i].grid()
# ax[0].set_ylabel(f'$\delta\phi$ [deg]')
# ax[1].set_ylabel(f'$\delta x$ [m]')
# ax[2].set_ylabel(f'$\delta y$ [m]')
# ax[2].set_xlabel('Timestep')
# fig.suptitle(f'Registration Errors')
# #ax[i].set_title(f'Error in SE2 State {i}')
# plt.tight_layout()
# plt.show()


## Violin plots 


# fig, ax = plt.subplots(3,1)

# for i in range(3):
#     ax[i].boxplot([error_sq_icp[:, i+2], error_sq_fou[:, i+2], error_sq_dir[:, i+2], error_sq_rad[:, i+2], error_sq_teas[:, i+2]],
#                showmeans=True)
#     ax[i].set_xticks([1, 2, 3, 4, 5])
#     ax[i].set_xticklabels([r'ICP', r'Fourier', r'Direct', r'Radon', r'TEASER'])
# ax[0].set_ylabel(r'$\delta\phi \, [\mathrm{{rad}}]$')
# ax[1].set_ylabel(r'$\delta x \,[\mathrm{{m}}]$')
# ax[2].set_ylabel(r'$\delta y \, [\mathrm{{m}}]$')
# ax[2].set_xlabel(r'Method')
# ax[0].set_title(r'Registration errors on OceanSim segment')
# plt.tight_layout()
# plt.show()

import seaborn as sns
import pandas as pd
sns.set_theme(style="whitegrid", palette="colorblind")

# methods = ['ICP', 'Fourier', 'Direct', 'Radon', 'TEASER']
# method_errors = [icp_skips, fou_skips, dir_skips, rad_skips, teas_skips]
# fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# for i in range(3):  # for each error dimension (e.g., phi, x, y)
#     data = []
#     positions = []
#     width = 0.13
#     for j, method in enumerate(methods):
#         # Collect the i-th error for each skip for this method
#         method_data = [errors[:, i+2] for errors in method_errors[j]]
#         # Flatten and collect for violinplot
#         for k, skip in enumerate(skips):
#             data.append(method_data[k])
#             positions.append(k + j*width - width*2)  # center groups

#     axes[i].boxplot(data, positions=positions, widths=width, showmeans=True)
#     # Set x-ticks at skip positions
#     axes[i].set_xticks(range(len(skips)))
#     axes[i].set_xticklabels([str(s) for s in skips])
#     axes[i].set_ylabel(['$\delta\phi$', '$\delta x$', '$\delta y$'][i])
#     axes[i].set_title(['Angle error', 'X error', 'Y error'][i])

# axes[-1].set_xlabel('Skip amount')
# plt.tight_layout()
# plt.show()

# import pandas as pd
# # Prepare data in long-form DataFrame
# records = []
# for method, method_name in zip(method_errors, methods):
#     for skip, errors in zip(skips, method):
#         for val in errors[:, 2]:  # for phi, change index for x/y
#             records.append({'Method': method_name, 'Skip': skip, 'Error': val})

# df = pd.DataFrame(records)
# sns.boxplot(x='Skip', y='Error', hue='Method', data=df)
# plt.show()

methods = ['ICP', 'FT', 'Direct', 'RT']
method_errors = [icp_skips, fou_skips, dir_skips, rad_skips ]
error_labels = [r'$\delta\alpha$', r'$\delta x$', r'$\delta y$']
ylabels = [r'$\delta\alpha \, [\mathrm{rad}]$', r'$\delta x \, [\mathrm{m}]$', r'$\delta y \, [\mathrm{m}]$']

fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

for i in range(3):  # 0: yaw, 1: x, 2: y
    records = []
    all_vals = []
    for method, method_name in zip(method_errors, methods):
        for skip, errors in zip(skips, method):
            vals = errors[:, 2 + i]  # 2: yaw, 3: x, 4: y
            all_vals.append(vals)

            for val in errors[:, 2 + i]:  # 2: yaw, 3: x, 4: y
                records.append({'Method': method_name, 'Skip': skip, 'Error': val})
    df = pd.DataFrame(records)
    sns.boxplot(x='Skip', y='Error', hue='Method', data=df, ax=axes[i], width=0.8, showfliers=False)
    axes[i].set_ylabel(ylabels[i])
    # axes[i].set_title(f'{error_labels[i]} error')
    if i == 0:
        handles, labels = axes[i].get_legend_handles_labels()
    axes[i].legend_.remove()  # Remove individual legends

    # Compute the largest 4th quartile (75th percentile) across all data for this error type
#     q3 = max(np.percentile(vals, 95) for vals in all_vals)
#     axes[i].set_ylim(top=q3, bottom=-0.0001)  # Set the same y-axis limit for all subplots

# axes[0].set_ylim(top= 0.5, bottom=-0.0001)  # Set the same y-axis limit for all subplots
# axes[1].set_ylim(top= 0.3, bottom=-0.0001)  # Set the same y-axis limit for all subplots
# axes[2].set_ylim(top= 0.9, bottom=-0.0001)  # Set the same y-axis limit for all subplots
# axes[0].legend(loc='upper right')
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, .97),ncol=len(methods), frameon=True)
# fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, 1.02))
axes[-1].set_xlabel(r'Frames skipped between registrations')


# fig.legend(handles, labels, loc='upper center', ncol=len(methods), bbox_to_anchor=(0.5, 1.02))
# axes[-1].set_xlabel(r'Frames skipped between registrations')

# fig.suptitle(r'Registration errors on real data')
plt.tight_layout(rect=[0, 0, 1, 0.93])  # Leave space at the top for the legend
plt.show()