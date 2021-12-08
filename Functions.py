import pyrosm
import pandas as pd
import numpy as np
from pyrosm import OSM, get_data
import geopandas as gpd
from matplotlib import pyplot as plt
import time
import geoplot
from shapely.geometry import Point
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import contextily as ctx
from cartopy.io import shapereader
from shapely.ops import unary_union
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import glob
import os

from geovoronoi.plotting import subplot_for_map, plot_voronoi_polys_with_points_in_area
from geovoronoi import voronoi_regions_from_coords, points_to_coords


def getElectricity(regions, crs, download_directory, update, filtered_data_directory):
    """

    :param regions:
    :return:
    """

    for i in regions:
        location = i
        print("Getting data for", i)
        fp = get_data(i, directory=download_directory, update=update)
        osm = OSM(fp)
        # Electricity towers
        towers_filter = {"power": ["tower"]}
        print("Filtering data for", i)
        towers = osm.get_data_by_custom_criteria(custom_filter=towers_filter, keep_ways=False, keep_relations=False)
        print("saving data for", i, "towers")
        name_towers = fr"{filtered_data_directory}{i}_towers.gpkg"
        towers.to_file(name_towers, driver="GPKG")
        # Electricity substations
        substation_filter = {"power": ["substation"]}
        print("Filtering data for", i)
        # substations_nodes = osm.get_data_by_custom_criteria(custom_filter=substation_filter, keep_ways=False, keep_relations=False)
        substations_polygons = osm.get_data_by_custom_criteria(custom_filter=substation_filter, keep_nodes=False, keep_relations=False)
        substations_poly_centroids = substations_polygons.copy()
        #substations_nodes.to_crs(crs)
        substations_poly_centroids = substations_poly_centroids.to_crs(crs)
        #substations_poly_centroids = substations_poly_centroids.explode()
        substations_poly_centroids['geometry'] = substations_poly_centroids['geometry'].centroid
        # substations = substations_nodes.append(substations_poly_centroids)
        # print("substations for:", i, substations)
        print("saving data for", i, "substation")
        name_substations = fr"{filtered_data_directory}{i}_substations.gpkg"
        substations_poly_centroids.to_file(name_substations, driver="GPKG")

    return()


def mergeTowers(regions, country, filtered_data_directory, country_directory):
    """

    :param regions:
    :return:
    """
    print("Merging towers datasets into simgle dataset")
    merged_towers = []

    for i in regions:
        region = i
        data = gpd.read_file(filtered_data_directory + i+"_towers.gpkg")
        merged_towers.append(data)
    towers = gpd.GeoDataFrame(pd.concat(merged_towers, ignore_index=True), crs=merged_towers[0].crs)
    towers['tower_ID'] = np.arange(towers.shape[0])
    towers.to_file(country_directory + country + "_towers.gpkg", driver="GPKG")


    return(towers)


def mergeSubstations(regions, country, filtered_data_directory, country_directory):
    """

    :param regions:
    :return:
    """
    print("Merging substation datasets into simgle dataset")
    merged_substations = []

    for i in regions:
        region = i
        data = gpd.read_file(filtered_data_directory + i+"_substations.gpkg")
        merged_substations.append(data)
    substations = gpd.GeoDataFrame(pd.concat(merged_substations, ignore_index=True), crs=merged_substations[0].crs)
    substations.to_file(country_directory + country + "_substations.gpkg", driver="GPKG")

    return(substations)


def filterGlobalPowerPlants(country, GlobalPowerPlants, crs, wgs, country_directory):
    Powerplant_data = GlobalPowerPlants
    Filtered_powerplants = Powerplant_data[Powerplant_data['country_long'] == country]
    Filtered_powerplants.to_csv(country_directory + country + "_powerplants.csv", index=False)
    filtered_powerplants_csv = pd.read_csv(country_directory + country + "_powerplants.csv")
    powerplants = gpd.GeoDataFrame(filtered_powerplants_csv, geometry=gpd.points_from_xy(
        Filtered_powerplants.longitude, Filtered_powerplants.latitude))
    powerplants = powerplants.set_crs(wgs)
    powerplants = powerplants.to_crs(crs)
    powerplants.to_file(country_directory + country +"_powerplants.gpkg", driver="GPKG")

    return(powerplants)


def convertCRS(powerplants, substations, towers, crs, grid):
    """

    :param powerplants:
    :param substations:
    :param towers:
    :param crs:
    :return:
    """
    substations.to_crs(crs)
    # powerplants = powerplants.set_crs(crs)
    # powerplants = powerplants.to_crs(crs)
    # print(powerplants.crs)
    towers = towers.to_crs(crs)

    return(substations, powerplants, towers, grid)


def getImportance (powerplants, substations, towers, grid, crs, country, country_directory):
    """
    :param powerplants:
    :param substations:
    :param towers:
    :param grid:
    :return:
    """
    # powerplants = powerplants.set_crs(crs)
    print("converting power plants to crs")
    powerplants = powerplants.to_crs(crs)
    print("converting towers to crs")
    towers = towers.to_crs(crs)
    print("converting substations to crs")
    substations = substations.to_crs(crs)
    #powerplants
    print("Starting powerplants")
    joined_powerplants_df = gpd.sjoin(powerplants, grid, how='inner', op='intersects')
    summed_capacity = joined_powerplants_df.groupby(['Grid_ID'], as_index=False)['estimated_generation_gwh'].sum()
    grid_powerplants = grid.merge(summed_capacity, on='Grid_ID', how='left')
    grid_powerplants['estimated_generation_gwh'].fillna(0, inplace=True)
    print("Powerplants complete")

    #towers
    print("Starting towers")
    joined_towers_df = gpd.sjoin(towers, grid, how='inner', op='intersects')
    count_towers = joined_towers_df.groupby(['Grid_ID'], as_index=False)['tower_ID'].count()
    count_towers.columns = ['Grid_ID', 'tower_count']
    grid_powerplants_towers = grid.merge(count_towers, on='Grid_ID', how='left')
    print("Towers complete")

    #substations
    print("Starting substations")
    country_boundary = gpd.read_file(r"G:\My Drive\Projects\EoS\Japan_electricity\Points_test\Japan_outline.gpkg")
    print("Line 169")
    country_boundary = country_boundary.to_crs(3395)
    print("Line 171")
    gdf_proj = substations.to_crs(country_boundary.crs)
    print("Line 173")
    gdf_proj = gpd.clip(gdf_proj, country_boundary)
    print("Line 175")
    boundary_shape = unary_union(country_boundary.geometry) #country_boundary.iloc[0].geometry
    print("Line 177")
    coords = points_to_coords(gdf_proj.geometry)
    print("Line 179")
    poly_shapes, poly_to_pt_assignments = voronoi_regions_from_coords(coords, boundary_shape) #pts,
    print("Line 181")
    voronoi_polygons =gpd.GeoDataFrame(gpd.GeoSeries(poly_shapes))
    print("Line 183")
    voronoi_polygons = voronoi_polygons.rename(columns={0:'geometry'}).set_geometry('geometry')
    print("Line 185")
    voronoi_polygons['sub_ID'] = np.arange(voronoi_polygons.shape[0])
    print("Line 187")
    voronoi_polygons = voronoi_polygons.set_crs(3395)
    print("Line 189")
    voronoi_polygons = voronoi_polygons.to_crs(crs)
    #voronoi_polygons.to_file("voronoi_polygons.gpkg", driver="GPKG")
    #Get population data
    population = gpd.read_file(
        r"G:\My Drive\Projects\EoS\Japan_electricity\Points_test\Japan_pop_2020_1km_aggregated.shp")
    # total_pop_test = population['VALUE'].sum()
    # print("Total population test: ", total_pop_test)
    joined_population_df = gpd.sjoin(population, voronoi_polygons, how='right', op='intersects')
    summed_population = joined_population_df.groupby(['sub_ID'], as_index=False)['VALUE'].sum()
    summed_population_test = summed_population['VALUE'].sum()
    merged_population = voronoi_polygons.merge(summed_population, on="sub_ID", how='left')
    #summed_population.columns = ['sub_ID', 'pop', 'geometry']
    joined_summed_population = gpd.sjoin(merged_population, substations, how='right', op='intersects') # test this
    summed_population_test = joined_summed_population['VALUE'].sum()

    #joined_summed_population.columns = ['geometry', 'sub_ID', 'VALUE', 'other_index', 'power', 'substation', 'id', 'timestamp', 'version', 'tags', 'osm_type']
    joined_summed_population = joined_summed_population.drop(['index_left'], axis=1)
    joined_summed_population_grid = gpd.sjoin(joined_summed_population, grid, how='right', op='within')
    summed_population_test = joined_summed_population_grid['VALUE'].sum()

    #joined_summed_population_grid_summed = joined_summed_population_grid['VALUE'].sum()
    joined_summed_population_grid_summed = joined_summed_population_grid.groupby(['Grid_ID'], as_index=False)['VALUE'].sum()
    grid_substations = grid.merge(joined_summed_population_grid_summed, on='Grid_ID', how='left')
    grid_substations['VALUE'].fillna(0, inplace=True)
    #print("Grid substations", grid_substations)

    total_capacity = grid_powerplants['estimated_generation_gwh'].sum()
    grid_powerplants['national_capacity_percentage'] = (grid_powerplants['estimated_generation_gwh']/total_capacity)*100
    total_towers = grid_powerplants_towers['tower_count'].sum()
    grid_powerplants_towers['tower_percentage'] = (grid_powerplants_towers['tower_count']/total_towers)*100
    total_population = grid_substations['VALUE'].sum()
    grid_substations['pop_percentage'] = (grid_substations['VALUE']/total_population)*100
    print("substations complete")

    print("Starting combining datasets")
    Combined_elec_1 = grid_substations.merge(grid_powerplants[['Grid_ID', 'national_capacity_percentage']], on='Grid_ID', how='left')
    Combined_elec_2 = Combined_elec_1.merge(grid_powerplants_towers[['Grid_ID', 'tower_percentage']],
                                on='Grid_ID', how='left')
    Combined_elec_2['importance_score'] = Combined_elec_2['national_capacity_percentage'] + Combined_elec_2['tower_percentage'] + Combined_elec_2['pop_percentage']
    #print(Combined_elec_2)
    importance = Combined_elec_2
    importance['importance_score'] = importance['importance_score'].fillna(0)
    importance['tower_percentage'] = importance['tower_percentage'].fillna(0)
    importance.drop('fid', inplace=True, axis=1)
    print("Combined complete, saving now")
    importance.to_file(country_directory  + country + "_importance_grid.gpkg", driver="GPKG")



    # volcanoes = pd.read_csv(r"C:\Users\hayes.jlee\Google Drive\Projects\EoS\Japan_electricity/Japan_volcanoes_FM_A1_fixed_unknowns.csv")
    # #print(volcanoes)
    # volcanoes_gdf = gpd.GeoDataFrame(volcanoes, geometry=gpd.points_from_xy(
    #     volcanoes.Longitude, volcanoes.Latitude))
    # #print(volcanoes_gdf)
    # volcanoes_gdf = volcanoes_gdf.set_crs(wgs)
    # volcanoes_gdf = volcanoes_gdf.to_crs(crs)
    # print("THE CRS OF VOLCANOES", volcanoes_gdf.crs)
    # print("volcanoes crs", volcanoes_gdf)
    # volcanoes_gdf['geometry'] = volcanoes_gdf.geometry.buffer(200000)
    # #volcanoes_gdf_buffer = gpd.GeoDataFrame(gpd.GeoSeries(volcanoes_gdf_buffer))
    # #volcanoes_gdf_buffer = volcanoes_gdf_buffer.rename(columns={0: 'geometry'}).set_geometry('geometry')
    # volcanoes_gdf_buffer_grid = gpd.sjoin(volcanoes_gdf, grid, how='inner', op='intersects')
    # volcanoes_gdf_buffer_grid_summed = volcanoes_gdf_buffer_grid.groupby(['grid_ID'], as_index=False)['Probability of eruption 50th percentile'].sum()
    # combined_volcano_elec = Combined_elec_2.merge(volcanoes_gdf_buffer_grid_summed[['grid_ID', 'Probability of eruption 50th percentile']], on='grid_ID', how='left')
    # combined_volcano_elec['Probability of eruption 50th percentile'].fillna(0, inplace=True)
    # combined_volcano_elec['disruption'] = combined_volcano_elec['Probability of eruption 50th percentile'] * combined_volcano_elec['combined_score']
    # #print(combined_volcano_elec)
    #
    # volcanoes_gdf.to_file("Volcano_buffer.gpkg", driver="GPKG")
    # combined_volcano_elec.to_file("combined_elec_grid_test.gpkg", driver="GPKG")
    #
    # # Combined_elec_2.to_file("total_combined_elec.gpkg", driver="GPKG")
    # # grid_substations.to_file("grid_substations_pop.gpkg", driver="GPKG")
    # # grid_powerplants.to_file("grid_powerplants_capacity", driver="GPKG")
    # # grid_powerplants_towers.to_file("grid_towers", driver="GPKG")
    #
    #
    # # grid_substations = grid_substations.drop(['left_x', 'right_x', 'bottom_x', 'top_x', 'geometry_x', 'geometry_y',
    # #                                           'other_index', 'id_left', 'timestamp', 'version', 'tags', 'osm_type',
    # #                                           'index_right', 'id_right', 'left_y', 'top_y', 'right_y', 'bottom_y'], axis=1)
    #
    # # geo_list = grid_substations['geometry'].tolist()
    # # grid_substations = gpd.GeoDataFrame(grid_substations, crs=crs, geometry=geo_list)


    return(importance)


def GetTephraProbOutputs (files, save_path):
    files = glob.glob(r"C:\Users\hayes.jlee\PycharmProjects\TephraProb-master\RUNS\VEI_7\4\SUM\all\COL\*.txt")
    #VEI = ['VEI_3']
    for f in files:
        string = f[-8:-4]
        print(string)
        data = pd.read_csv(f, sep="\t", header=None)
        data.columns = ["x", "y", "loading"]
        save_path = f"Model_run_{string}.csv"
        save_file = data.to_csv(r"C:\Users\hayes.jlee\Desktop/VEI7" + "/" + save_path, index=False)
        #print(data)
    return()


def getImpact(powerplants, substations, towers, Hazard_0_1kg,Hazard_1kg,Hazard_10kg,Hazard_100kg, Hazard_1000kg, volcanoes, grid, crs):
    """

    :param powerplants:
    :param substations:
    :param towers:
    :param grid:
    :return:
    """
    powerplants = powerplants.to_crs(crs)
    towers = towers.to_crs(crs)
    substations = substations.to_crs(crs)

    VEI3_ash = glob.glob(r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra\Fujisan\VEI3\*.csv")
    VEI4_ash = glob.glob(
        r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra\Fujisan\VEI4\*.csv")
    VEI5_ash = glob.glob(
        r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra\Fujisan\VEI5\*.csv")
    VEI6_ash = glob.glob(
        r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra\Fujisan\VEI6\*.csv")
    VEI7_ash = glob.glob(
        r"G:\My Drive\Projects\EoS\Japan_electricity\Tephra_thickness\Processed_tephra\Fujisan\VEI7\*.csv")

    run = 0
    for footprint in VEI3_ash:
        grid_thickness = pd.merge(grid, footprint, how='outer')
        grid_thickness['thickness'] = grid_thickness['thickness'].fillna(0)
        run +=1
        col_name = f"thickness_{run}"
        grid_thickness.rename(columns={"thickness": col_name})
        powerplants_impact = gpd.sjoin(powerplants, grid_thickness, how='left', op='interects')
        #powerplants_impact['IS1'] = powerplants_impact[col_name].apply(lambda x: 0.113 * powerplants_impact[col_name] if (x < 5) else)

        if powerplants_impact[col_name] < 5:
            powerplants_impact['IS1'] = 0.113 * powerplants_impact[col_name]
        elif powerplants_impact[col_name] >= 5 and powerplants_impact[col_name] < 30.5:
            powerplants_impact['IS1']= 0.008 * powerplants_impact[col_name] + 0.527
        elif powerplants_impact[col_name] >= 30.5 and powerplants_impact[col_name] < 300:
            powerplants_impact['IS1'] = 0.001 * powerplants_impact[col_name] + 0.741
        elif powerplants_impact[col_name] > 300:
            powerplants_impact['IS1'] = 1





    hazard = [Hazard_0_1kg, Hazard_1kg, Hazard_10kg, Hazard_100kg, Hazard_1000kg]
    for num in range(5):
        for h in hazard:
            ash_hazard = h
            powerplants_impact = gpd.sjoin(powerplants, ash_hazard, how='left', op='intersects')
            substations_impact = gpd.sjoin(substations, ash_hazard, how='left', op='intersects')
            towers_impact = gpd.sjoin(towers, ash_hazard, how='left', op='intersects')

            if num == 1:

                # # # Powerplants # # #

                # # Conditional probability of impact state for powerplants @ 0.1 kg/m2
                powerplants_impact['Cond_Exceed_IS0'] = 0.2* 0.1
                powerplants_impact['Cond_Exceed_IS1'] = 0.07 * 0.1
                powerplants_impact['Cond_Exceed_IS2'] = 0.02 * 0.1
                powerplants_impact['Cond_Exceed_IS3'] = 0.004 * 0.1

                # # Absolute probabililty of impact state for powerplants @ 0.1 kg/m2 for all volcanoes
                powerplants_impact['National_Exceed_IS0'] = powerplants_impact['Cond_Exceed_IS0'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS1'] = powerplants_impact['Cond_Exceed_IS1'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS2'] = powerplants_impact['Cond_Exceed_IS2'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS3'] = powerplants_impact['Cond_Exceed_IS3'] * powerplants_impact['summed_total_probability']

                # # Absolute probabililty of impact state for powerplants @ 0.1 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    powerplants_impact[volcano_label_IS0] = powerplants_impact['Cond_Exceed_IS0'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS1] = powerplants_impact['Cond_Exceed_IS1'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS2] = powerplants_impact['Cond_Exceed_IS2'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS3] = powerplants_impact['Cond_Exceed_IS3'] * powerplants_impact[volcano_probability]

                # # # substations # # #

                # # Conditional probability of impact state for substations @ 0.1 kg/m2
                substations_impact['Cond_Exceed_IS0'] = 0.2 * 0.1
                substations_impact['Cond_Exceed_IS1'] = 0.14 * 0.1
                substations_impact['Cond_Exceed_IS2'] = 0.03 * 0.1
                substations_impact['Cond_Exceed_IS3'] = 0.01 * 0.1

                # # Absolute probabililty of impact state for substations @ 0.1 kg/m2 for all volcanoes
                substations_impact['National_Exceed_IS0'] = substations_impact['Cond_Exceed_IS0'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS1'] = substations_impact['Cond_Exceed_IS1'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS2'] = substations_impact['Cond_Exceed_IS2'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS3'] = substations_impact['Cond_Exceed_IS3'] * substations_impact['summed_total_probability']

                # # Absolute probabililty of impact state for substations @ 0.1 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    substations_impact[volcano_label_IS0] = substations_impact['Cond_Exceed_IS0'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS1] = substations_impact['Cond_Exceed_IS1'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS2] = substations_impact['Cond_Exceed_IS2'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS3] = substations_impact['Cond_Exceed_IS3'] * substations_impact[volcano_probability]

                # # # substations # # #

                # # Conditional probability of impact state for transmission towers @ 0.1 kg/m2
                towers_impact['Cond_Exceed_IS0'] = 0.2 * 0.1
                towers_impact['Cond_Exceed_IS1'] = 0.14 * 0.1
                towers_impact['Cond_Exceed_IS2'] = 0.03 * 0.1
                towers_impact['Cond_Exceed_IS3'] = 0.01 * 0.1

                # # Absolute probabililty of impact state for transmission towers @ 0.1 kg/m2 for all volcanoes
                towers_impact['National_Exceed_IS0'] = towers_impact['Cond_Exceed_IS0'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS1'] = towers_impact['Cond_Exceed_IS1'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS2'] = towers_impact['Cond_Exceed_IS2'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS3'] = towers_impact['Cond_Exceed_IS3'] * towers_impact['summed_total_probability']

                # # Absolute probabililty of impact state for transmission towers @ 0.1 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    towers_impact[volcano_label_IS0] = towers_impact['Cond_Exceed_IS0'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS1] = towers_impact['Cond_Exceed_IS1'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS2] = towers_impact['Cond_Exceed_IS2'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS3] = towers_impact['Cond_Exceed_IS3'] * towers_impact[volcano_probability]

            if num == 2:

                # # # Powerplants # # #

                # # Conditional probability of impact state for powerplants @ 1 kg/m2
                powerplants_impact['Cond_Exceed_IS0'] = 0.2 * 1
                powerplants_impact['Cond_Exceed_IS1'] = 0.07 * 1
                powerplants_impact['Cond_Exceed_IS2'] = 0.02 * 1
                powerplants_impact['Cond_Exceed_IS3'] = 0.004 * 1

                # # Absolute probabililty of impact state for powerplants @ 1 kg/m2 for all volcanoes
                powerplants_impact['National_Exceed_IS0'] = powerplants_impact['Cond_Exceed_IS0'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS1'] = powerplants_impact['Cond_Exceed_IS1'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS2'] = powerplants_impact['Cond_Exceed_IS2'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS3'] = powerplants_impact['Cond_Exceed_IS3'] * powerplants_impact['summed_total_probability']

                # # Absolute probabililty of impact state for powerplants @ 1 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    powerplants_impact[volcano_label_IS0] = powerplants_impact['Cond_Exceed_IS0'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS1] = powerplants_impact['Cond_Exceed_IS1'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS2] = powerplants_impact['Cond_Exceed_IS2'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS3] = powerplants_impact['Cond_Exceed_IS3'] * powerplants_impact[volcano_probability]

                # # # Substatiosn # # #
                # # Conditional probability of impact state for substations @ 1 kg/m2
                substations_impact['Cond_Exceed_IS0'] = 0.2 * 1
                substations_impact['Cond_Exceed_IS1'] = 0.14 * 1
                substations_impact['Cond_Exceed_IS2'] = 0.03 * 1
                substations_impact['Cond_Exceed_IS3'] = 0.01 * 1

                # # Absolute probabililty of impact state for substations @ 1 kg/m2 for all volcanoes
                substations_impact['National_Exceed_IS0'] = substations_impact['Cond_Exceed_IS0'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS1'] = substations_impact['Cond_Exceed_IS1'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS2'] = substations_impact['Cond_Exceed_IS2'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS3'] = substations_impact['Cond_Exceed_IS3'] * substations_impact['summed_total_probability']

                # # Absolute probabililty of impact state for substations @ 1 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    substations_impact[volcano_label_IS0] = substations_impact['Cond_Exceed_IS0'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS1] = substations_impact['Cond_Exceed_IS1'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS2] = substations_impact['Cond_Exceed_IS2'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS3] = substations_impact['Cond_Exceed_IS3'] * substations_impact[volcano_probability]

                # # # Transmission towers # # #
                # # Conditional probability of impact state for transmission towers @ 1 kg/m2
                towers_impact['Cond_Exceed_IS0'] = 0.2 * 1
                towers_impact['Cond_Exceed_IS1'] = 0.14 * 1
                towers_impact['Cond_Exceed_IS2'] = 0.03 * 1
                towers_impact['Cond_Exceed_IS3'] = 0.01 * 1

                # # Absolute probabililty of impact state for transmission towers @ 1 kg/m2 for all volcanoes
                towers_impact['National_Exceed_IS0'] = towers_impact['Cond_Exceed_IS0'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS1'] = towers_impact['Cond_Exceed_IS1'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS2'] = towers_impact['Cond_Exceed_IS2'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS3'] = towers_impact['Cond_Exceed_IS3'] * towers_impact['summed_total_probability']

                # # Absolute probabililty of impact state for transmission towers @ 1 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    towers_impact[volcano_label_IS0] = towers_impact['Cond_Exceed_IS0'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS1] = towers_impact['Cond_Exceed_IS1'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS2] = towers_impact['Cond_Exceed_IS2'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS3] = towers_impact['Cond_Exceed_IS3'] * towers_impact[volcano_probability]


            if num == 3:

                # # # Powerplnts # # #

                # # Conditional probability of impact state for powerplants @ 10 kg/m2
                powerplants_impact['Cond_Exceed_IS0'] = 1
                powerplants_impact['Cond_Exceed_IS1'] = (0.016 *10) + 0.272
                powerplants_impact['Cond_Exceed_IS2'] = (0.011 *10) + 0.046
                powerplants_impact['Cond_Exceed_IS3'] = (0.001 *10) + 0.014

                # # Absolute probabililty of impact state for powerplants @ 10 kg/m2 for all volcanoes
                powerplants_impact['National_Exceed_IS0'] = powerplants_impact['Cond_Exceed_IS0'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS1'] = powerplants_impact['Cond_Exceed_IS1'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS2'] = powerplants_impact['Cond_Exceed_IS2'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS3'] = powerplants_impact['Cond_Exceed_IS3'] * powerplants_impact['summed_total_probability']

                # # Absolute probabililty of impact state for powerplants @ 10 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    powerplants_impact[volcano_label_IS0] = powerplants_impact['Cond_Exceed_IS0'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS1] = powerplants_impact['Cond_Exceed_IS1'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS2] = powerplants_impact['Cond_Exceed_IS2'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS3] = powerplants_impact['Cond_Exceed_IS3'] * powerplants_impact[volcano_probability]

                # # # Substations # # #

                # # Conditional probability of impact state for substations @ 10 kg/m2
                substations_impact['Cond_Exceed_IS0'] = 1
                substations_impact['Cond_Exceed_IS1'] = (0.004 *10) + 0.680
                substations_impact['Cond_Exceed_IS2'] = (0.01 * 10) + 0.101
                substations_impact['Cond_Exceed_IS3'] = (0.002 *10) + 0.04

                # # Absolute probabililty of impact state for substations @ 10 kg/m2 for all volcanoes
                substations_impact['National_Exceed_IS0'] = substations_impact['Cond_Exceed_IS0'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS1'] = substations_impact['Cond_Exceed_IS1'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS2'] = substations_impact['Cond_Exceed_IS2'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS3'] = substations_impact['Cond_Exceed_IS3'] * substations_impact['summed_total_probability']

                # # Absolute probabililty of impact state for substations @ 10 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    substations_impact[volcano_label_IS0] = substations_impact['Cond_Exceed_IS0'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS1] = substations_impact['Cond_Exceed_IS1'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS2] = substations_impact['Cond_Exceed_IS2'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS3] = substations_impact['Cond_Exceed_IS3'] * substations_impact[volcano_probability]

                # # # Transmission towers # # #

                # # Conditional probability of impact state for transmission towers @ 10 kg/m2
                towers_impact['Cond_Exceed_IS0'] = 1
                towers_impact['Cond_Exceed_IS1'] = (0.004 * 10) + 0.680
                towers_impact['Cond_Exceed_IS2'] = (0.01 * 10) + 0.101
                towers_impact['Cond_Exceed_IS3'] = (0.002 * 10) + 0.04

                # # Absolute probabililty of impact state for transmission towers  @ 10 kg/m2 for all volcanoes
                towers_impact['National_Exceed_IS0'] = towers_impact['Cond_Exceed_IS0'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS1'] = towers_impact['Cond_Exceed_IS1'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS2'] = towers_impact['Cond_Exceed_IS2'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS3'] = towers_impact['Cond_Exceed_IS3'] * towers_impact['summed_total_probability']

                # # Absolute probabililty of impact state for transmission towers @ 10 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    towers_impact[volcano_label_IS0] = towers_impact['Cond_Exceed_IS0'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS1] = towers_impact['Cond_Exceed_IS1'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS2] = towers_impact['Cond_Exceed_IS2'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS3] = towers_impact['Cond_Exceed_IS3'] * towers_impact[volcano_probability]

            if num == 4:

                # # # Powerplants # # #

                # # Conditional probability of impact state for powerplants @ 100 kg/m2
                powerplants_impact['Cond_Exceed_IS0'] = 1
                powerplants_impact['Cond_Exceed_IS1'] = (0.001*100)+0.71
                powerplants_impact['Cond_Exceed_IS2'] = (0.001*100)+0.349
                powerplants_impact['Cond_Exceed_IS3'] = (0.0003*100)+0.039

                # # Absolute probabililty of impact state for powerplants @ 100 kg/m2 for all volcanoes
                powerplants_impact['National_Exceed_IS0'] = powerplants_impact['Cond_Exceed_IS0'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS1'] = powerplants_impact['Cond_Exceed_IS1'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS2'] = powerplants_impact['Cond_Exceed_IS2'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS3'] = powerplants_impact['Cond_Exceed_IS3'] * powerplants_impact['summed_total_probability']

                # # Absolute probabililty of impact state for powerplants @ 100 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    powerplants_impact[volcano_label_IS0] = powerplants_impact['Cond_Exceed_IS0'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS1] = powerplants_impact['Cond_Exceed_IS1'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS2] = powerplants_impact['Cond_Exceed_IS2'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS3] = powerplants_impact['Cond_Exceed_IS3'] * powerplants_impact[volcano_probability]

                # # # Substations # # #
                # Conditional probability of impact state for substations @ 100 kg/m2
                substations_impact['Cond_Exceed_IS0'] = 1
                substations_impact['Cond_Exceed_IS1'] = (0.0003 *100) + 0.789
                substations_impact['Cond_Exceed_IS2'] = (0.001 *100) + 0.379
                substations_impact['Cond_Exceed_IS3'] = (0.001 *100) + 0.079

                # # Absolute probabililty of impact state for substations @ 100 kg/m2 for all volcanoes
                substations_impact['National_Exceed_IS0'] = substations_impact['Cond_Exceed_IS0'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS1'] = substations_impact['Cond_Exceed_IS1'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS2'] = substations_impact['Cond_Exceed_IS2'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS3'] = substations_impact['Cond_Exceed_IS3'] * substations_impact['summed_total_probability']

                # # Absolute probabililty of impact state for substations @ 100 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    substations_impact[volcano_label_IS0] = substations_impact['Cond_Exceed_IS0'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS1] = substations_impact['Cond_Exceed_IS1'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS2] = substations_impact['Cond_Exceed_IS2'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS3] = substations_impact['Cond_Exceed_IS3'] * substations_impact[volcano_probability]

                # # # Transmission towers # # #
                # Conditional probability of impact state for transmission towers @ 100 kg/m2
                towers_impact['Cond_Exceed_IS0'] = 1
                towers_impact['Cond_Exceed_IS1'] = (0.0003 * 100) + 0.789
                towers_impact['Cond_Exceed_IS2'] = (0.001 * 100) + 0.379
                towers_impact['Cond_Exceed_IS3'] = (0.001 * 100) + 0.079

                # # Absolute probabililty of impact state for transmission towers @ 100 kg/m2 for all volcanoes
                towers_impact['National_Exceed_IS0'] = towers_impact['Cond_Exceed_IS0'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS1'] = towers_impact['Cond_Exceed_IS1'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS2'] = towers_impact['Cond_Exceed_IS2'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS3'] = towers_impact['Cond_Exceed_IS3'] * towers_impact['summed_total_probability']

                # # Absolute probabililty of impact state for transmission towers @ 100 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    towers_impact[volcano_label_IS0] = towers_impact['Cond_Exceed_IS0'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS1] = towers_impact['Cond_Exceed_IS1'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS2] = towers_impact['Cond_Exceed_IS2'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS3] = towers_impact['Cond_Exceed_IS3'] * towers_impact[volcano_probability]

            if num == 5:

                # # # Powerplants # # #
                # Conditional probability of impact state for powerplants @ 1000 kg/m2
                powerplants_impact['Cond_Exceed_IS0'] = 1
                powerplants_impact['Cond_Exceed_IS1'] = (0.001*1000)+0.71
                powerplants_impact['Cond_Exceed_IS2'] = (0.001*1000)+0.349
                powerplants_impact['Cond_Exceed_IS3'] = (0.0003*1000)+0.039

                # # Absolute probabililty of impact state for powerplants @ 100 kg/m2 for all volcanoes
                powerplants_impact['National_Exceed_IS0'] = powerplants_impact['Cond_Exceed_IS0'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS1'] = powerplants_impact['Cond_Exceed_IS1'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS2'] = powerplants_impact['Cond_Exceed_IS2'] * powerplants_impact['summed_total_probability']
                powerplants_impact['National_Exceed_IS3'] = powerplants_impact['Cond_Exceed_IS3'] * powerplants_impact['summed_total_probability']

                # # Absolute probabililty of impact state for powerplants @ 100 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    powerplants_impact[volcano_label_IS0] = powerplants_impact['Cond_Exceed_IS0'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS1] = powerplants_impact['Cond_Exceed_IS1'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS2] = powerplants_impact['Cond_Exceed_IS2'] * powerplants_impact[volcano_probability]
                    powerplants_impact[volcano_label_IS3] = powerplants_impact['Cond_Exceed_IS3'] * powerplants_impact[volcano_probability]

                # # # Substations # # #
                # Conditional probability of impact state for substations @ 1000 kg/m2
                substations_impact['Cond_Exceed_IS0'] = 1
                substations_impact['Cond_Exceed_IS1'] = (0.0003 *1000) + 0.789
                substations_impact['Cond_Exceed_IS2'] = (0.001 *1000) + 0.379
                substations_impact['Cond_Exceed_IS3'] = (0.001 *1000) + 0.079

                # # Absolute probabililty of impact state for substations @ 100 kg/m2 for all volcanoes
                substations_impact['National_Exceed_IS0'] = substations_impact['Cond_Exceed_IS0'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS1'] = substations_impact['Cond_Exceed_IS1'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS2'] = substations_impact['Cond_Exceed_IS2'] * substations_impact['summed_total_probability']
                substations_impact['National_Exceed_IS3'] = substations_impact['Cond_Exceed_IS3'] * substations_impact['summed_total_probability']

                # # Absolute probabililty of impact state for substations @ 100 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    substations_impact[volcano_label_IS0] = substations_impact['Cond_Exceed_IS0'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS1] = substations_impact['Cond_Exceed_IS1'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS2] = substations_impact['Cond_Exceed_IS2'] * substations_impact[volcano_probability]
                    substations_impact[volcano_label_IS3] = substations_impact['Cond_Exceed_IS3'] * substations_impact[volcano_probability]


                # # # Transmission towers # # #
                # Conditional probability of impact state for transmission towers @ 1000 kg/m2
                towers_impact['Cond_Exceed_IS0'] = 1
                towers_impact['Cond_Exceed_IS1'] = (0.0003 * 1000) + 0.789
                towers_impact['Cond_Exceed_IS2'] = (0.001 * 1000) + 0.379
                towers_impact['Cond_Exceed_IS3'] = (0.001 * 1000) + 0.079

                # # Absolute probabililty of impact state for transmission towers @ 100 kg/m2 for all volcanoes
                towers_impact['National_Exceed_IS0'] = towers_impact['Cond_Exceed_IS0'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS1'] = towers_impact['Cond_Exceed_IS1'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS2'] = towers_impact['Cond_Exceed_IS2'] * towers_impact['summed_total_probability']
                towers_impact['National_Exceed_IS3'] = towers_impact['Cond_Exceed_IS3'] * towers_impact['summed_total_probability']

                # # Absolute probabililty of impact state for transmission towers @ 100 kg/m2 per volcano
                for v in volcanoes:
                    volcano_name = volcanoes['Volcano Na']
                    volcano_probability = "summed_" + volcano_name + "_probability"
                    volcano_label_IS0 = volcano_name + "_Exceed_IS0"
                    volcano_label_IS1 = volcano_name + "_Exceed_IS1"
                    volcano_label_IS2 = volcano_name + "_Exceed_IS2"
                    volcano_label_IS3 = volcano_name + "_Exceed_IS3"
                    towers_impact[volcano_label_IS0] = towers_impact['Cond_Exceed_IS0'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS1] = towers_impact['Cond_Exceed_IS1'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS2] = towers_impact['Cond_Exceed_IS2'] * towers_impact[volcano_probability]
                    towers_impact[volcano_label_IS3] = towers_impact['Cond_Exceed_IS3'] * towers_impact[volcano_probability]


    powerplants_impact['PP_impact_score'] = powerplants_impact['Cond_Exceed_IS1']*1000
    powerplants_impact = powerplants_impact[['geometry',
                                             'Cond_Exceed_IS0',
                                             'Cond_Exceed_IS1',
                                             'Cond_Exceed_IS2',
                                             'Cond_Exceed_IS3',
                                             'PP_impact_score']]

    joined_powerplants_vuln_df = gpd.sjoin(powerplants_impact, grid, how='inner', op='intersects')
    summed_pp_vuln = joined_powerplants_vuln_df.groupby(['grid_ID'], as_index=False)['PP_impact_score'].sum()
    grid_powerplants_vuln = grid.merge(summed_pp_vuln, on='grid_ID', how='left')
    grid_powerplants_vuln['PP_impact_score'].fillna(0, inplace=True)

    substations_impact['SS_impact_score'] = substations_impact['Exceed_IS1'] * 100
    substations_impact = substations_impact[['geometry', 'SS_impact_score']]

    joined_substations_vuln_df = gpd.sjoin(substations_impact, grid, how='inner', op='intersects')
    summed_ss_vuln = joined_substations_vuln_df.groupby(['grid_ID'], as_index=False)['SS_impact_score'].sum()
    grid_substations_vuln = grid.merge(summed_ss_vuln, on='grid_ID', how='left')
    grid_substations_vuln['SS_impact_score'].fillna(0, inplace=True)

    towers_impact['TT_impact_score'] = towers_impact['Exceed_IS1'] * 1
    towers_impact = towers_impact[['geometry', 'TT_impact_score']]

    joined_towers_vuln_df = gpd.sjoin(towers_impact, grid, how='inner', op='intersects')
    summed_tt_vuln = joined_towers_vuln_df.groupby(['grid_ID'], as_index=False)['TT_impact_score'].sum()
    grid_towers_vuln = grid.merge(summed_tt_vuln, on='grid_ID', how='left')
    grid_towers_vuln['TT_impact_score'].fillna(0, inplace=True)

    impact_PP_TT = grid_powerplants_vuln.merge(grid_towers_vuln, on="grid_ID", how='left')
    impact = impact_PP_TT.merge(grid_substations_vuln, on="grid_ID", how='left')
    impact['impact_score'] = impact['PP_impact_score'] + impact['TT_impact_score'] + impact['SS_impact_score']
    impact['impact_score'].fillna(0, inplace=True)
    impact = impact[['geometry', 'impact_score', 'grid_ID']]
    impact_gdf = gpd.GeoDataFrame(impact)
    impact_gdf.to_file("testing_impact_fuji.shp")

    return(impact)


def getDisruption(scenario, impact, importance):
    """

    :param impact:
    :param importance:
    :return:
    """
    print("Importance") #### no grid_id on importance####
    impact = impact[['impact_score', 'grid_ID']]
    impact_score_total = impact['impact_score'].sum()
    print("impact score: ", impact_score_total)
    #importance = importance.rename(columns={'id': 'grid_ID'})
    importance = importance[['Importance', 'grid_ID', 'geometry']]
    score = impact.merge(importance, on='grid_ID', how='inner')
    #score['impact_score'].fillna(0, inplace=True)
    impact_score_total = score['impact_score'].sum()
    print("impact score: ", impact_score_total)
    score['disruption'] = score['Importance'] * score['impact_score']
    disruption_score = score['disruption'].sum()
    print("disruption_score is: ", disruption_score)
    score_gdf = gpd.GeoDataFrame(score)
    #score_gdf.to_file("testing_disruption_fuji.shp")
    #disruption_file = score.to_file(r"C:\Users\hayes.jlee\Google Drive\Projects\EoS\Japan_electricity\case_studies\shinmoedake/", scenario, "_disruption_grid.shp")

    return()

def getMap(geospatial_data, volcanoes):
    from contextily.tile import warp_img_transform, warp_tiles, _warper

    reproject = geospatial_data.to_crs(epsg=4326)
    reproject_volcanoes = volcanoes.to_crs(epsg=4326)

    fig, ax = plt.subplots(figsize=(12,8))
    ax = reproject.plot(ax=ax, column="Importance",
                    linewidth=0.03,
                    cmap="Purples",
                    alpha=0.5,
                    legend=True
                    )
    ctx.add_basemap(ax, crs=reproject_volcanoes.crs.to_string(), source=ctx.providers.Stamen.TonerLite)
    ctx.add_basemap(ax, crs=reproject_volcanoes.crs.to_string(), source=ctx.providers.Stamen.TonerLabels)
    # ax.get_legend().set_bbox_to_anchor((1.3, 0.22))
    # ax.get_legend().set_title("Importance score")
    reproject_volcanoes.plot(ax=ax, marker="^", facecolor="black", edgecolor="white", linewidth=0.1)
    plt.grid(b=True, which='major', color='#666666', linestyle='-', linewidth=0.1)
    # scalebar = AnchoredSizeBar(ax.transData, 1000000, "1000 km", 1, pad=0.1)
    # ax.add_artist(scalebar)
    plt.tight_layout()
    plt.show()

    return()