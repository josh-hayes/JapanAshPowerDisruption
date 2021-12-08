import pandas as pd
import numpy as np
import geopandas as gpd
import time
import glob
import os
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns

from Functions import getElectricity
from Functions import mergeSubstations
from Functions import mergeTowers
from Functions import filterGlobalPowerPlants
from Functions import convertCRS
from Functions import getImportance
from Functions import GetTephraProbOutputs
from Functions import getImpact
from Functions import getDisruption
from Functions import getMap

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 2000)
start_time = time.time()



grid = gpd.read_file(r"G:\My Drive\Projects\EoS\Japan_electricity\volcanoes\Japan_10km_grid_spatial_index.shp")
#VEI3_ash = glob.glob(r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra\Fujisan\VEI3\*.csv")
powerplants = gpd.read_file("Japan_powerplants")
towers = gpd.read_file("Japan_towers")
towers = towers.to_crs(crs=3857)
substations = gpd.read_file("Japan_substations")
VEI3_ash_test = pd.read_csv(r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra\Fujisan\VEI3\Fujisan_VEI3_198.csv")


A = 5
B = 30.5
C = 300
run = 0
ash_footprints = r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra\Fujisan\VEI7\*.csv"
#
def getImpactScore (run, A, B, C, ash_footprints):

    for filename in glob.glob(ash_footprints):
        run +=1
        footprint = pd.read_csv(filename)
        print("Running analysis for model run: ", run)
        powerplants_join = gpd.sjoin(powerplants, grid, how='left', op='intersects')
        powerplants_thickness = pd.merge(powerplants_join, footprint, how='left')
        powerplants_thickness['thickness'] = powerplants_thickness['thickness'].fillna(0)
        powerplants_thickness = powerplants_thickness[powerplants_thickness.thickness != 0]

        powerplants_conditions = [(powerplants_thickness['thickness'] < A),
                      (powerplants_thickness['thickness'] >= A) & (powerplants_thickness['thickness'] < B),
                      (powerplants_thickness['thickness'] >= B) & (powerplants_thickness['thickness'] < C),
                      powerplants_thickness['thickness'] > C]

        powerplant_IS1_equations = [0.113 * powerplants_thickness['thickness'],
                                0.008 * powerplants_thickness['thickness'] + 0.527,
                                0.001 * powerplants_thickness['thickness'] + 0.741,
                                1]

        powerplant_IS2_equations = [0.043 * powerplants_thickness['thickness'],
                                    0.009 * powerplants_thickness['thickness'] + 0.171,
                                    0.001 * powerplants_thickness['thickness'] + 0.415,
                                    1]

        #note: there are no documented IS3 powerplants in the world.
        powerplant_IS3_equations = [0.013 * powerplants_thickness['thickness'],
                                0.003 * powerplants_thickness['thickness'] + 0.052,
                                0.001 * powerplants_thickness['thickness'] + 0.128,
                                0.001*300+0.128]

        powerplants_thickness['IS1'] = np.select(powerplants_conditions, powerplant_IS1_equations, default=0)
        powerplants_thickness['IS1'].values[powerplants_thickness['IS1'] > 1] = 1
        powerplants_thickness['IS2'] = np.select(powerplants_conditions, powerplant_IS2_equations, default=0)
        powerplants_thickness['IS2'].values[powerplants_thickness['IS2'] > 1] = 1
        powerplants_thickness['IS3'] = np.select(powerplants_conditions, powerplant_IS3_equations, default=0)
        powerplants_thickness['IS3'].values[powerplants_thickness['IS3'] > 1] = 1

        powerplants_impact = powerplants_thickness.groupby(['Grid_ID']).sum().reset_index()

        powerplants_impact = powerplants_impact[['Grid_ID', 'IS1', 'IS2', 'IS3']]
        powerplants_impact = powerplants_impact.rename(columns={"IS1": "PP_IS1", "IS2": "PP_IS2", "IS3": "PP_IS3"})

        # powerplants_impact.to_csv("powerplants_VEI3_fuji_test.csv")

        #substations
        substations_join = gpd.sjoin(substations, grid, how='left', op='intersects')
        substations_thickness = pd.merge(substations_join, footprint, how='left')
        substations_thickness['thickness'] = substations_thickness['thickness'].fillna(0)
        substations_thickness = substations_thickness[substations_thickness.thickness != 0]

        substations_conditions = [(substations_thickness['thickness'] < A),
                      (substations_thickness['thickness'] >= A) & (substations_thickness['thickness'] < B),
                      (substations_thickness['thickness'] >= B) & (substations_thickness['thickness'] < C),
                      substations_thickness['thickness'] > C]

        substations_IS1_equations = [0.14 * substations_thickness['thickness'],
                                0.004 * substations_thickness['thickness'] + 0.680,
                                0.0003 * substations_thickness['thickness'] + 0.789,
                                1]

        substations_IS2_equations = [0.03 * substations_thickness['thickness'],
                                    0.01 * substations_thickness['thickness'] + 0.101,
                                    0.001 * substations_thickness['thickness'] + 0.379,
                                    1]

        substations_IS3_equations = [0.01 * substations_thickness['thickness'],
                                0.002 * substations_thickness['thickness'] + 0.04,
                                0.001 * substations_thickness['thickness'] + 0.079,
                                0.001*300+0.079]

        substations_thickness['IS1'] = np.select(substations_conditions, substations_IS1_equations, default=0)
        substations_thickness['IS1'].values[substations_thickness['IS1'] > 1] = 1
        substations_thickness['IS2'] = np.select(substations_conditions, substations_IS2_equations, default=0)
        substations_thickness['IS2'].values[substations_thickness['IS2'] > 1] = 1
        substations_thickness['IS3'] = np.select(substations_conditions, substations_IS3_equations, default=0)
        substations_thickness['IS3'].values[substations_thickness['IS3'] > 1] = 1

        substations_impact = substations_thickness.groupby(['Grid_ID']).sum().reset_index()

        substations_impact = substations_impact[['Grid_ID', 'IS1', 'IS2', 'IS3']]
        substations_impact = substations_impact.rename(columns={"IS1": "SS_IS1", "IS2": "SS_IS2", "IS3": "SS_IS3"})

        # substations_impact.to_csv("substations_VEI3_fuji_test.csv")


        #Transmission towers
        towers_join = gpd.sjoin(towers, grid, how='left', op='intersects')
        towers_thickness = pd.merge(towers_join, footprint, how='left')
        towers_thickness['thickness'] = towers_thickness['thickness'].fillna(0)
        towers_thickness = towers_thickness[towers_thickness.thickness != 0]


        towers_conditions = [(towers_thickness['thickness'] < A),
                      (towers_thickness['thickness'] >= A) & (towers_thickness['thickness'] < B),
                      (towers_thickness['thickness'] >= B) & (towers_thickness['thickness'] < C),
                      towers_thickness['thickness'] > C]

        towers_IS1_equations = [0.14 * towers_thickness['thickness'],
                                0.004 * towers_thickness['thickness'] + 0.680,
                                0.0003 * towers_thickness['thickness'] + 0.789,
                                1]

        towers_IS2_equations = [0.03 * towers_thickness['thickness'],
                                    0.01 * towers_thickness['thickness'] + 0.101,
                                    0.001 * towers_thickness['thickness'] + 0.379,
                                    1]

        towers_IS3_equations = [0.01 * towers_thickness['thickness'],
                                0.002 * towers_thickness['thickness'] + 0.04,
                                0.001 * towers_thickness['thickness'] + 0.079,
                                0.001*300+0.079]

        towers_thickness['IS1'] = np.select(towers_conditions, towers_IS1_equations, default=0)
        towers_thickness['IS1'].values[towers_thickness['IS1'] > 1] = 1
        towers_thickness['IS2'] = np.select(towers_conditions, towers_IS2_equations, default=0)
        towers_thickness['IS2'].values[towers_thickness['IS2'] > 1] = 1
        towers_thickness['IS3'] = np.select(towers_conditions, towers_IS3_equations, default=0)
        towers_thickness['IS3'].values[towers_thickness['IS3'] > 1] = 1

        towers_impact = towers_thickness.groupby(['Grid_ID']).sum().reset_index()

        towers_impact = towers_impact[['Grid_ID', 'IS1', 'IS2', 'IS3']]
        towers_impact = towers_impact.rename(columns={"IS1": "TT_IS1", "IS2": "TT_IS2", "IS3": "TT_IS3"})

        Combined = pd.merge(towers_impact, powerplants_impact, how='outer')
        Combined = pd.merge(Combined, substations_impact, how='outer')
        Combined = Combined.fillna(0)
        Combined['IS1'] = Combined['PP_IS1'] + Combined['TT_IS1'] + Combined['SS_IS1']
        Combined['IS2'] = Combined['PP_IS2'] + Combined['TT_IS2'] + Combined['SS_IS2']
        Combined['IS3'] = Combined['PP_IS3'] + Combined['TT_IS3'] + Combined['SS_IS3']
        Combined['Impact_score'] = Combined['IS1'] + Combined['IS2'] + Combined['IS3']
        save_name = f"combined_VEI3_{run}.csv"

        Combined.to_csv(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\VEI7" + "/" + save_name)
#
def weightbyfrequency(run):
    VEI_dict = ['VEI3', 'VEI4', 'VEI5', 'VEI6', 'VEI7']
    volcano = "Fujisan"
    annual_probabilities = pd.read_csv(r"Japan_volcano_probabilities.csv")
    for v in VEI_dict:
        VEI = v
        print("Running analysis for:", v)
        run = 0
        #file_location = f"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\{VEI}"
        for filename in glob.glob(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm" + "/" + VEI + "/*.csv"):
            run += 1
            print("running analysis for model run:", run)
            impact_footprint = pd.read_csv(filename)
            probabtility = annual_probabilities[VEI].values[0]
            impact_footprint['Impact_score_weighted'] = impact_footprint['Impact_score'] * probabtility
            save_name = f"{volcano}_combined_{VEI}_{run}.csv"
            impact_footprint.to_csv(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm"+ "/" + VEI + "/" + save_name)
#
# # run_model = weightbyfrequency(run)

def normaliseImportance():
    importance = gpd.read_file(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Japan_importance_grid.gpkg")
    column = importance['importance_score']
    max_value = column.max()
    print("Max value:", max_value)
    min_value = column.min()
    print("Min value:", min_value)
    importance['importance_score_normalised'] = ((importance['importance_score'] - min_value)/max_value)*10
    importance.to_file(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Japan_importance_grid_normalised.gpkg", driver="GPKG")
    return()

# normalised_importance = normaliseImportance()

def combineImportanceImpact():
    importance = gpd.read_file(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Japan_importance_grid_normalised.gpkg")
    impact = pd.read_csv(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Fujisan_overall_impact.csv")
    impact = impact[['Grid_ID', 'Impact_score_normalised']]
    merged_files = pd.merge(importance, impact, how='left')
    merged_files['disruption'] = merged_files['importance_score_normalised'] * merged_files['Impact_score_normalised']
    merged_files.to_file(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Fujisan_disruption.gpkg",
                       driver="GPKG")
    return()
#
# disruption = combineImportanceImpact()

def getFMDisruption():
    importance = gpd.read_file(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Japan_importance_grid_normalised.gpkg")
    VEI_dict = ['VEI3', 'VEI4', 'VEI5', 'VEI6', 'VEI7']
    volcano = "Fujisan"
    path = r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Fujisan"
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    column = concatenated_df['Impact_score_weighted']
    max_value = column.max()
    print("max value:", max_value)
    min_value = column.min()
    print("min value:", min_value)
    #concatenated_df['Impact_score_normalised'] = ((concatenated_df['Impact_score_weighted'] - min_value)/max_value)*10
    run = 0
    disruption_score = []
    model_run = []
    for filename in glob.glob(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Fujisan\*.csv"):
        run += 1
        print("running analysis for model run:", run)
        impact_footprint = pd.read_csv(filename)
        combined = pd.merge(importance, impact_footprint, how='left')
        combined['Impact_score_normalised'] = ((combined['Impact_score_weighted'] - min_value)/max_value)*10
        combined['disruption'] = combined['importance_score_normalised'] * combined['Impact_score_normalised']
        total_disruption = combined['disruption'].sum()
        print("total footprint disruption:", total_disruption)
        disruption_score.append(total_disruption)
        model_run.append(run)
    df = pd.DataFrame({'model run': model_run, 'disruption score': disruption_score})
    df.to_csv(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm"+ "/disruption_scores.csv")
    bins = 40
    sns.displot(disruption_score, bins=bins, kde=False)
    plt.ylabel('Frequency')
    plt.xlabel('Footprint disruption score')
    plt.show()

    return()
#
# footprint_disruption = getFMDisruption()

def getGIFMAP():
    from pathlib import Path
    import contextily as cx
    output_path = r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\maps"
    Japan_grid = gpd.read_file(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Japan_importance_grid_normalised.gpkg")
    study_extent = gpd.read_file(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Fuji_study_area.gpkg")
    vmin = 1
    vmax = 1000
    VEI = ['VEI5'] #['VEI3', 'VEI4', 'VEI5', 'VEI6', 'VEI7']
    for v in VEI:
        size = v
        run = 0
        tephra_path = f"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra\Fujisan\{size}\*csv"
        for filename in glob.glob(tephra_path):
            run += 1
            print("running analysis for model run:", run)
            name = Path(filename).stem
            impact_footprint = pd.read_csv(filename)
            merged = pd.merge(Japan_grid, impact_footprint, how='left')
            merged['thickness'] = merged['thickness'].fillna(0)
            merged = merged[merged['thickness'] > 1]
            merged.to_crs(epsg=3857)
            bounds = study_extent.total_bounds
            xlim = ([bounds[0], bounds[2]])
            ylim = ([bounds[1], bounds[3]])
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax = merged.plot(column=merged['thickness'], cmap='Blues', figsize=(10, 10), alpha=0.6, linewidth=0.8, edgecolor='0.8', vmin=vmin,
                               vmax=vmax,
                               legend=True, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.axis('off')
            cx.add_basemap(ax, source=cx.providers.Stamen.TerrainBackground)
            fig.tight_layout()
            #cx.add_basemap(ax, source=cx.providers.Stamen.TonerLabels)
            #plt.show()
            filepath = os.path.join(output_path, name + '.jpg')
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            chart = fig.get_figure()
            plt.show()
            # chart.savefig(filepath, dpi=72)
            # if run == 50:
            #     break

    import imageio
    image_path = r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\maps"
    all_files = glob.glob(os.path.join(image_path, "*.jpg"))
    images = []
    for jpg in all_files:
        images.append(imageio.imread(jpg))
    imageio.mimsave(r'C:\Users\hayes.jlee\PycharmProjects\Pyrosm\VEI5_basemap.gif', images)

    from pygifsicle import optimize
    optimize(r'C:\Users\hayes.jlee\PycharmProjects\Pyrosm\VEI5_basemap.gif')
    return()

make_map = getGIFMAP()

def aggImpacts():
    path = r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Fujisan"
    all_files = glob.glob(os.path.join(path, "*.csv"))
    df_from_each_file = (pd.read_csv(f) for f in all_files)
    concatenated_df = pd.concat(df_from_each_file, ignore_index=True)
    overall_impact = concatenated_df.groupby(["Grid_ID"]).sum().reset_index()
    overall_impact['Impact_score_weighted'] = overall_impact['Impact_score_weighted'].fillna(0)
    column = overall_impact['Impact_score_weighted']
    max_value = column.max()
    print("max value:", max_value)
    min_value = column.min()
    print("min value:", min_value)
    overall_impact['Impact_score_normalised'] = ((overall_impact['Impact_score_weighted'] - min_value)/max_value)*10
    overall_impact.to_csv(r"C:\Users\hayes.jlee\PycharmProjects\Pyrosm\Fujisan_overall_impact.csv")
    return()

aggregated_impacts = aggImpacts()



# run = 0
# for footprint in VEI3_ash:
#     grid_thickness = pd.merge(grid, footprint, how='outer')
#     grid_thickness['thickness'] = grid_thickness['thickness'].fillna(0)
#     run +=1
#     col_name = f"thickness_{run}"
#     grid_thickness.rename(columns={"thickness": col_name})


# country = "Japan"
# regions = ['shikoku', 'chubu', 'chugoku', 'hokkaido', 'kansai', 'kanto', 'kyushu', 'tohoku']
# crs = "EPSG:3857"
# wgs = "EPSG:4326"
# grid = gpd.read_file(r"G:\My Drive\Projects\EoS\Japan_electricity\volcanoes\Japan_10km_grid_spatial_index.shp")
# grid = grid.to_crs(crs)
# powerplants = gpd.read_file("Japan_powerplants")
# towers = gpd.read_file("Japan_towers")
# substations = gpd.read_file("Japan_substations")
#
# #Calculate importance value
# importance = getImportance(powerplants, substations, towers, grid, crs)
# importance = importance.drop(['fid'], axis=1)
# importance.to_file("Japan_importance_grid.gpkg", driver="GPKG")

# Calculate impact value


# Volcanoes = ["Fujisan"]
# VEI = ["VEI3", "VEI4", "VEI5", "VEI6", "VEI7"]

# for v in Volcanoes:
#     volcano_name = v
#     hazard_path_VEI3 = r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra" + "/" + volcano_name + "/VEI3" + "/*.csv"
#     hazard_path_VEI4 = r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra" + "/" + volcano_name + "/VEI4" + "/*.csv"
#     hazard_path_VEI5 = r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra" + "/" + volcano_name + "/VEI5" + "/*.csv"
#     hazard_path_VEI6 = r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra" + "/" + volcano_name + "/VEI6" + "/*.csv"
#     hazard_path_VEI7 = r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra" + "/" + volcano_name + "/VEI7" + "/*.csv"
#
# ash_hazard_VEI3 = glob.glob(hazard_path_VEI3)
# ash_hazard_VEI4 = glob.glob(hazard_path_VEI4)
# ash_hazard_VEI5 = glob.glob(hazard_path_VEI5)
# ash_hazard_VEI6 = glob.glob(hazard_path_VEI6)
# ash_hazard_VEI7 = glob.glob(hazard_path_VEI7)



#grid['grid_ID'] = np.arange(grid.shape[0])

#GlobalPowerPlants = pd.read_csv(r"C:\Users\hayes.jlee\Google Drive\Projects\EoS\Exposure\data\global-power-plant-database-master\output_database/global_power_plant_database.csv")







###########
# volcanoes = ['Volcano 1', 'Volcano 2', 'Volcano 3']
#
# for volcano_name in volcanoes:
#     for VEI in range (3, 8):
#         for run in range(1, 5001):
#             path = f"tephra_outputs\{volcano_name}\{VEI}\{run}\VEI_{VEI}_0001.csv"
#             print(path)
#################









# # print('downloading data')
# # download_data = getElectricity(regions)
# print("merging towers")
# mergetowers = mergeTowers(regions, country)
# print("merging substations")
# mergesubstations = mergeSubstations(regions, country)
# print("filtering power plants")
# filterpowerplants = filterGlobalPowerPlants(country, GlobalPowerPlants)
# print("converting crs")
# convertedcrs = convertCRS(filterpowerplants, mergesubstations, mergetowers, crs)
# powerplants = convertedcrs[1]
# substations = convertedcrs[0]
# towers = convertedcrs[2]
# grid = convertedcrs[3]
# print("getting importance value")
# Importance = getImportance(filterpowerplants, mergesubstations, mergetowers, grid, crs)
#
# print(f"Parsing substations and towers lasted {round(time.time() - start_time, 0)} seconds.")

# print("Getting hazard data")
# hazard = gpd.read_file(r"C:\Users\hayes.jlee\Google Drive\Projects\EoS\Japan_electricity\case_studies\Fuji_1707/Fuji_tephra_grid.shp")
# print("Getting importance grid")
# importance = gpd.read_file(r"C:\Users\hayes.jlee\Google Drive\Projects\EoS\Japan_electricity\case_studies\shinmoedake\importance_test.shp")
# print("Getting powerplants")
# powerplants = gpd.read_file(r"C:\Users\hayes.jlee\Google Drive\Projects\EoS\Japan_electricity\Points_test\Japan_power_plants.gpkg")
# print("Getting transmission towers")
# towers = gpd.read_file(r"C:\Users\hayes.jlee\Google Drive\Projects\EoS\Japan_electricity\Points_test\Towers_WGS_Pseudo\Japan_merged_towers.shp")
# print("Getting substations towers")
# substations = gpd.read_file(r"C:\Users\hayes.jlee\Google Drive\Projects\EoS\Japan_electricity\Points_test\Japan_substations.gpkg")
# print("Assessing impact")
# impact = getImpact(powerplants, substations, towers, hazard, grid)
# scenario = "Shinmoedake"
# print("Assessing disruption")
# shinmoedake_disruption = getDisruption(scenario, impact, importance)
# volcanoes = gpd.read_file(r"C:\Users\hayes.jlee\Google Drive\Projects\EoS\Japan_electricity\case_studies\Fuji_1707\asd.shp")
# print("Making map of importance")
# geospatial_data = importance
# show_map = getMap(geospatial_data, volcanoes)