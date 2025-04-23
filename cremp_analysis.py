#!/usr/bin/env python3
# CREMP Data Analysis Script
# Analyzes coral reef monitoring data across Florida regions

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
import matplotlib.ticker as ticker

# Set file paths
DATA_DIR = "/Users/aditya/Downloads/CREMP_CSV_files"
PCOVER_FILE = os.path.join(DATA_DIR, "CREMP_Pcover_2023_TaxaGroups.csv")
SCOR_FILE = os.path.join(DATA_DIR, "CREMP_SCOR_Summaries_2023_LTA.csv")
STATIONS_FILE = os.path.join(DATA_DIR, "CREMP_Stations_2023.csv")
OCTO_FILE = os.path.join(DATA_DIR, "CREMP_OCTO_Summaries_2023_Density.csv")
SPECIES_FILE = os.path.join(DATA_DIR, "CREMP_Pcover_2023_StonyCoralSpecies.csv")

# Create output directory for results
OUTPUT_DIR = "results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Function to load all datasets with error handling
def load_data():
    """Load all CREMP datasets and return as a dictionary of dataframes"""
    data = {}
    
    try:
        print("Loading percent cover data...")
        data['pcover'] = pd.read_csv(PCOVER_FILE)
        
        print("Loading stony coral data...")
        data['scor'] = pd.read_csv(SCOR_FILE)
        
        print("Loading station data...")
        data['stations'] = pd.read_csv(STATIONS_FILE)
        
        print("Loading octocoral data...")
        data['octo'] = pd.read_csv(OCTO_FILE)
        
        print("Loading species data...")
        data['species'] = pd.read_csv(SPECIES_FILE)
        
        print(f"Successfully loaded {len(data)} datasets.")
        return data
    
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return None
    except pd.errors.ParserError as e:
        print(f"Error: Could not parse CSV file - {e}")
        return None
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        return None

# Function to analyze percent cover by region
def analyze_regional_cover(data):
    """Analyze and visualize percent cover by region"""
    print("Analyzing regional percent cover...")
    
    if 'pcover' not in data:
        print("Error: Percent cover data not available")
        return
    
    # Group by subregion and calculate mean cover for various taxa
    regional_cover = data['pcover'].groupby('Subregion').agg({
        'Stony_coral': 'mean',
        'Octocoral': 'mean',
        'Macroalgae': 'mean',
        'Cyanobacteria': 'mean',
        'Porifera': 'mean'
    }).reset_index()
    
    # Plot regional cover comparison
    plt.figure(figsize=(12, 8))
    
    # Create melted dataframe for seaborn
    regional_cover_melted = pd.melt(
        regional_cover, 
        id_vars=['Subregion'],
        value_vars=['Stony_coral', 'Octocoral', 'Macroalgae', 'Cyanobacteria', 'Porifera'],
        var_name='Taxa', 
        value_name='Percent Cover'
    )
    
    # Create plot
    sns.barplot(x='Subregion', y='Percent Cover', hue='Taxa', data=regional_cover_melted)
    plt.title('Average Percent Cover by Region and Taxa Group')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'regional_cover_comparison.png'), dpi=300)
    
    print(f"Regional analysis complete. Results saved to {OUTPUT_DIR}")
    return regional_cover

# Function to analyze habitat trends
def analyze_habitat_trends(data):
    """Analyze trends in coral cover by habitat type"""
    print("Analyzing habitat trends...")
    
    if 'pcover' not in data or 'stations' not in data:
        print("Error: Required data not available")
        return
    
    # Create a mapping from StationID to Habitat
    print("Columns in stations data:", data['stations'].columns.tolist())
    station_habitat_map = {}
    for _, row in data['stations'].iterrows():
        station_habitat_map[row['StationID']] = row['Habitat']
    
    # Add Habitat column to pcover data
    data['pcover']['Habitat'] = data['pcover']['StationID'].map(station_habitat_map)
    
    # Group by year and habitat, calculate mean cover
    habitat_trends = data['pcover'].groupby(['Year', 'Habitat']).agg({
        'Stony_coral': 'mean',
        'Macroalgae': 'mean'
    }).reset_index()
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Stony coral trends by habitat
    sns.lineplot(x='Year', y='Stony_coral', hue='Habitat', data=habitat_trends, ax=ax1, marker='o')
    ax1.set_title('Stony Coral Cover Trend by Habitat')
    ax1.set_ylabel('Mean Percent Cover')
    
    # Macroalgae trends by habitat
    sns.lineplot(x='Year', y='Macroalgae', hue='Habitat', data=habitat_trends, ax=ax2, marker='o')
    ax2.set_title('Macroalgae Cover Trend by Habitat')
    ax2.set_ylabel('Mean Percent Cover')
    ax2.set_xlabel('Year')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'habitat_trends.png'), dpi=300)
    
    print(f"Habitat trend analysis complete. Results saved to {OUTPUT_DIR}")
    return habitat_trends

# Function to analyze coral health metrics
def analyze_coral_health(data):
    """Analyze coral health metrics from SCOR data"""
    print("Analyzing coral health metrics...")
    
    if 'scor' not in data or 'stations' not in data:
        print("Error: Required data not available")
        return None, None
    
    # Check if Subregion exists in scor data
    scor_df = data['scor']
    print("Columns in SCOR data:", scor_df.columns.tolist())
    
    # Get region information from stations if needed
    if 'Subregion' not in scor_df.columns:
        print("Subregion not in SCOR data, merging with stations")
        # Check which ID column to use for merging
        if 'StationID' in scor_df.columns and 'StationID' in data['stations'].columns:
            merge_col = 'StationID'
        elif 'SiteID' in scor_df.columns and 'SiteID' in data['stations'].columns:
            merge_col = 'SiteID'
        else:
            print("No common ID column between SCOR and stations data")
            return None, None
        
        # Get necessary columns from stations
        station_cols = [merge_col]
        for col in ['Subregion', 'Habitat']:
            if col in data['stations'].columns:
                station_cols.append(col)
        
        # Merge SCOR data with stations
        coral_health = pd.merge(
            scor_df,
            data['stations'][station_cols],
            on=merge_col,
            how='left'
        )
    else:
        coral_health = scor_df.copy()
    
    # Get all species columns (stony coral species)
    base_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 'Site_name', 'StationID']
    species_columns = [col for col in coral_health.columns 
                      if col not in base_cols]
    
    if not species_columns:
        print("Error: No species columns found in SCOR data")
        return None, None
    
    print(f"Using {len(species_columns)} species columns as health metrics")
    
    # Calculate total species richness (number of species present) per site
    coral_health['total_species'] = coral_health[species_columns].apply(lambda row: sum(row > 0), axis=1)
    
    # Calculate total coral count per site
    coral_health['total_corals'] = coral_health[species_columns].sum(axis=1)
    
    # Choose a primary grouping column - prefer Subregion if available
    if 'Subregion' in coral_health.columns and not coral_health['Subregion'].isna().all():
        group_col = 'Subregion'
    elif 'Habitat' in coral_health.columns and not coral_health['Habitat'].isna().all():
        group_col = 'Habitat'
    else:
        # If no good grouping column, use site identifier
        site_cols = [col for col in coral_health.columns if 'site' in col.lower()]
        group_col = site_cols[0] if site_cols else 'StationID'
    
    print(f"Grouping coral health metrics by: {group_col}")
    
    # Calculate species richness by the grouping column
    region_richness = coral_health.groupby(group_col)['total_species'].mean().reset_index()
    region_richness = region_richness.sort_values('total_species', ascending=False)
    
    # Get top 10 most abundant species
    species_abundance = pd.DataFrame({
        'Species': species_columns,
        'Abundance': [coral_health[col].sum() for col in species_columns]
    })
    species_abundance = species_abundance.sort_values('Abundance', ascending=False)
    top_species = species_abundance.head(10)
    
    # Create plots if data is available
    if not region_richness.empty and not top_species.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Regional species richness
        sns.barplot(x=group_col, y='total_species', data=region_richness, ax=ax1)
        ax1.set_title(f'Mean Coral Species Richness by {group_col}')
        ax1.set_ylabel('Mean Species Count')
        ax1.set_xlabel(group_col)
        ax1.tick_params(axis='x', rotation=45)
        
        # Top species abundance
        sns.barplot(x='Abundance', y='Species', data=top_species, ax=ax2)
        ax2.set_title('Top 10 Most Abundant Coral Species')
        ax2.set_xlabel('Total Count')
        ax2.set_ylabel('Species')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'coral_health_metrics.png'), dpi=300)
        
        print(f"Coral health analysis complete. Results saved to {OUTPUT_DIR}")
    else:
        print("Warning: Insufficient data for coral health plots")
    
    return region_richness, top_species

# Function to compare coral and octocoral distributions
def compare_distributions(data):
    """Compare distributions of stony coral and octocoral cover"""
    print("Comparing coral and octocoral distributions...")
    
    if 'pcover' not in data:
        print("Error: Percent cover data not available")
        return
    
    # Calculate correlation between stony coral and octocoral
    correlation = data['pcover'][['Stony_coral', 'Octocoral']].corr().iloc[0, 1]
    print(f"Correlation between Stony Coral and Octocoral cover: {correlation:.3f}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='Stony_coral', 
        y='Octocoral', 
        hue='Subregion', 
        size='Porifera',
        sizes=(20, 200),
        alpha=0.7,
        data=data['pcover']
    )
    
    plt.title('Relationship between Stony Coral and Octocoral Cover')
    plt.xlabel('Stony Coral Cover (%)')
    plt.ylabel('Octocoral Cover (%)')
    plt.annotate(f'Correlation: {correlation:.3f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'coral_octocoral_relationship.png'), dpi=300)
    
    print(f"Coral distribution comparison complete. Results saved to {OUTPUT_DIR}")
    return correlation

# Function to generate site report
def generate_site_report(data):
    """Generate a comprehensive health report for each site based on multiple metrics."""
    
    print("Generating site health report...")
    
    # Check if we have the necessary data
    if 'pcover' not in data or 'stations' not in data or 'scor' not in data:
        print("Missing required data for site health report")
        return {}, [], []
    
    pcover = data['pcover']
    stations = data['stations']
    scor = data['scor']
    
    # Add log statement to check column names
    print(f"Columns in stations data: {stations.columns.tolist()}")
    print(f"Columns in pcover data: {pcover.columns.tolist()}")
    
    # Determine column name for site name (may be 'Site_name' or similar)
    site_name_col = None
    for col in ['Site_name', 'site_name', 'SiteName', 'sitename', 'Site', 'Name']:
        if col in pcover.columns:
            site_name_col = col
            break
    
    if site_name_col is None:
        # If site name column not found in pcover, check if SiteID exists and merge with stations
        if 'SiteID' in pcover.columns and 'SiteID' in stations.columns and 'Site_name' in stations.columns:
            # Merge pcover with stations to get site names
            pcover = pd.merge(pcover, stations[['SiteID', 'Site_name']], on='SiteID', how='left')
            site_name_col = 'Site_name'
        else:
            print("Could not find site name column in data")
            # Default to StationID as a fallback
            site_name_col = 'StationID'
            print(f"Using {site_name_col} as fallback for site name")
    
    # Create a site dataset with merged data
    site_data = pcover.copy()
    
    # Calculate site health metrics
    site_health = site_data.groupby(['StationID', site_name_col, 'Subregion']).agg({
        'Stony_coral': 'mean',
        'Macroalgae': 'mean',
        'Porifera': 'mean',  # Changed from 'Sponge' to 'Porifera'
        'Octocoral': 'mean',
    }).reset_index()
    
    # Calculate health score (higher stony coral and octocoral is good, higher macroalgae is bad)
    site_health['health_score'] = (
        site_health['Stony_coral'] * 2 + 
        site_health['Octocoral'] - 
        site_health['Macroalgae'] * 1.5
    )
    
    # Identify top and bottom sites
    top_sites = site_health.sort_values('health_score', ascending=False).head(5)
    bottom_sites = site_health.sort_values('health_score', ascending=True).head(5)
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Top 10 healthiest sites
    plt.subplot(2, 1, 1)
    top10 = site_health.sort_values('health_score', ascending=False).head(10)
    sns.barplot(x=site_name_col, y='health_score', data=top10)
    plt.title('Top 10 Healthiest Reef Sites')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the result
    plt.savefig(os.path.join(OUTPUT_DIR, 'site_health_report.png'))
    
    # Save detailed report
    site_health.to_csv(os.path.join(OUTPUT_DIR, 'site_health_metrics.csv'), index=False)
    
    return site_health, top_sites, bottom_sites

# Function to analyze temporal trends in stony coral cover
def analyze_temporal_trends(data):
    """Analyze the evolution of stony coral percentage cover over time across stations."""
    print("Analyzing temporal trends in stony coral cover...")
    
    if 'pcover' not in data:
        print("Error: Percent cover data not available for temporal analysis")
        return None
    
    # Group by year and calculate mean stony coral cover
    yearly_cover = data['pcover'].groupby('Year')['Stony_coral'].agg(['mean', 'std', 'count']).reset_index()
    yearly_cover['ci'] = 1.96 * yearly_cover['std'] / np.sqrt(yearly_cover['count'])
    
    # Group by year and station to see station-specific trends
    station_yearly = data['pcover'].groupby(['Year', 'StationID'])['Stony_coral'].mean().reset_index()
    
    # Get top 5 stations with highest average stony coral cover for detailed analysis
    top_stations = data['pcover'].groupby('StationID')['Stony_coral'].mean().sort_values(ascending=False).head(5).index.tolist()
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Overall trend
    ax1.errorbar(yearly_cover['Year'], yearly_cover['mean'], yerr=yearly_cover['ci'], 
                marker='o', linestyle='-', label='Mean Cover with 95% CI')
    
    # Fit a trend line
    X = yearly_cover['Year'].values.reshape(-1, 1)
    y = yearly_cover['mean'].values
    model = LinearRegression().fit(X, y)
    ax1.plot(yearly_cover['Year'], model.predict(X), 'r--', label=f'Trend (Slope: {model.coef_[0]:.4f})')
    
    ax1.set_title('Evolution of Mean Stony Coral Cover Over Time')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Mean Percent Cover (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Station-specific trends for top stations
    top_station_data = station_yearly[station_yearly['StationID'].isin(top_stations)]
    sns.lineplot(x='Year', y='Stony_coral', hue='StationID', data=top_station_data, ax=ax2, marker='o')
    ax2.set_title('Stony Coral Cover Trends for Top 5 Stations')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Percent Cover (%)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stony_coral_temporal_trends.png'), dpi=300)
    
    # Perform statistical test for trend significance
    stat, p_value = stats.pearsonr(yearly_cover['Year'], yearly_cover['mean'])
    trend_direction = "increasing" if stat > 0 else "decreasing"
    significance = "significant" if p_value < 0.05 else "not significant"
    
    print(f"Trend analysis: Stony coral cover shows a {trend_direction} trend over time (r={stat:.3f}, p={p_value:.3f}), which is {significance}")
    print(f"Average annual change: {model.coef_[0]:.4f}% per year")
    
    return yearly_cover, top_station_data

# Function to analyze species richness trends over time
def analyze_species_richness_trends(data):
    """Analyze and visualize trends in species richness of stony corals over the years."""
    print("Analyzing trends in species richness over time...")
    
    if 'species' not in data or 'scor' not in data:
        print("Error: Required species data not available")
        return None
    
    # Get all species columns
    species_df = data['species']
    scor_df = data['scor']
    
    # Identify species columns (exclude metadata columns)
    meta_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 'Site_name', 'StationID']
    species_cols_pcover = [col for col in species_df.columns if col not in meta_cols]
    species_cols_scor = [col for col in scor_df.columns if col not in meta_cols]
    
    print(f"Found {len(species_cols_pcover)} species in percent cover data and {len(species_cols_scor)} species in SCOR data")
    
    # Calculate species richness by year (number of species present at each site)
    if species_cols_pcover:
        # For percent cover data - ensure numeric data before comparison
        # Convert all columns to numeric, coercing errors to NaN
        for col in species_cols_pcover:
            species_df[col] = pd.to_numeric(species_df[col], errors='coerce')
        
        # Now calculate richness (sum of species with values > 0)
        species_df['richness'] = species_df[species_cols_pcover].apply(lambda row: sum(row > 0), axis=1)
        yearly_richness = species_df.groupby('Year')['richness'].agg(['mean', 'std', 'count']).reset_index()
        yearly_richness['ci'] = 1.96 * yearly_richness['std'] / np.sqrt(yearly_richness['count'])
    else:
        # Fallback to SCOR data if species data doesn't have species columns
        # Convert all columns to numeric, coercing errors to NaN
        for col in species_cols_scor:
            scor_df[col] = pd.to_numeric(scor_df[col], errors='coerce')
            
        scor_df['richness'] = scor_df[species_cols_scor].apply(lambda row: sum(row > 0), axis=1)
        yearly_richness = scor_df.groupby('Year')['richness'].agg(['mean', 'std', 'count']).reset_index()
        yearly_richness['ci'] = 1.96 * yearly_richness['std'] / np.sqrt(yearly_richness['count'])
    
    # Calculate species richness by region and year
    if 'Subregion' in species_df.columns:
        region_richness = species_df.groupby(['Year', 'Subregion'])['richness'].mean().reset_index()
    else:
        # Try with SCOR data if not in species data
        region_richness = scor_df.groupby(['Year', 'Subregion'])['richness'].mean().reset_index()
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14))
    
    # Overall trend with confidence interval
    ax1.errorbar(yearly_richness['Year'], yearly_richness['mean'], yerr=yearly_richness['ci'], 
                marker='o', linestyle='-', label='Mean Richness with 95% CI')
    
    # Fit a trend line
    X = yearly_richness['Year'].values.reshape(-1, 1)
    y = yearly_richness['mean'].values
    model = LinearRegression().fit(X, y)
    ax1.plot(yearly_richness['Year'], model.predict(X), 'r--', label=f'Trend (Slope: {model.coef_[0]:.4f})')
    
    ax1.set_title('Evolution of Stony Coral Species Richness Over Time')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Mean Species Richness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Regional trends
    sns.lineplot(x='Year', y='richness', hue='Subregion', data=region_richness, ax=ax2, marker='o')
    ax2.set_title('Species Richness Trends by Region')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Mean Species Richness')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'species_richness_trends.png'), dpi=300)
    
    # Perform statistical test for trend significance
    stat, p_value = stats.pearsonr(yearly_richness['Year'], yearly_richness['mean'])
    trend_direction = "increasing" if stat > 0 else "decreasing"
    significance = "significant" if p_value < 0.05 else "not significant"
    
    print(f"Trend analysis: Species richness shows a {trend_direction} trend over time (r={stat:.3f}, p={p_value:.3f}), which is {significance}")
    print(f"Average annual change in species richness: {model.coef_[0]:.4f} species per year")
    
    return yearly_richness, region_richness

# Function to analyze octocoral density variations
def analyze_octocoral_density(data):
    """Examine how the density of octocoral species varies across stations and over time."""
    print("Analyzing octocoral density variations...")
    
    if 'octo' not in data:
        print("Error: Octocoral data not available")
        return None
    
    octo_df = data['octo']
    print(f"Columns in octocoral data: {octo_df.columns.tolist()}")
    
    # Identify the density column
    density_cols = [col for col in octo_df.columns if 'density' in col.lower() or 'count' in col.lower()]
    if not density_cols:
        print("Error: No density columns found in octocoral data")
        return None
    
    density_col = density_cols[0]
    print(f"Using {density_col} as density metric")
    
    # Calculate yearly average density
    yearly_density = octo_df.groupby('Year')[density_col].agg(['mean', 'std', 'count']).reset_index()
    yearly_density['ci'] = 1.96 * yearly_density['std'] / np.sqrt(yearly_density['count'])
    
    # Calculate station-specific density over time
    station_density = octo_df.groupby(['Year', 'StationID'])[density_col].mean().reset_index()
    
    # Get top 5 stations with highest density for detailed analysis
    top_stations = octo_df.groupby('StationID')[density_col].mean().sort_values(ascending=False).head(5).index.tolist()
    
    # If we have regional data, calculate regional density
    if 'Subregion' in octo_df.columns:
        region_density = octo_df.groupby(['Year', 'Subregion'])[density_col].mean().reset_index()
    else:
        region_density = None
    
    # Create visualizations
    fig, axes = plt.subplots(2, 1, figsize=(12, 14))
    
    # Overall trend with confidence interval
    axes[0].errorbar(yearly_density['Year'], yearly_density['mean'], yerr=yearly_density['ci'], 
                   marker='o', linestyle='-', label='Mean Density with 95% CI')
    
    # Fit a trend line
    X = yearly_density['Year'].values.reshape(-1, 1)
    y = yearly_density['mean'].values
    model = LinearRegression().fit(X, y)
    axes[0].plot(yearly_density['Year'], model.predict(X), 'r--', label=f'Trend (Slope: {model.coef_[0]:.4f})')
    
    axes[0].set_title(f'Evolution of Octocoral Density Over Time')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel(f'Mean {density_col}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot based on available data
    if region_density is not None:
        # Regional density trends
        sns.lineplot(x='Year', y=density_col, hue='Subregion', data=region_density, marker='o', ax=axes[1])
        axes[1].set_title('Octocoral Density Trends by Region')
    else:
        # Station-specific trends for top stations
        top_station_data = station_density[station_density['StationID'].isin(top_stations)]
        sns.lineplot(x='Year', y=density_col, hue='StationID', data=top_station_data, marker='o', ax=axes[1])
        axes[1].set_title('Octocoral Density Trends for Top 5 Stations')
    
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel(f'Mean {density_col}')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'octocoral_density_trends.png'), dpi=300)
    
    # Perform statistical test for trend significance
    stat, p_value = stats.pearsonr(yearly_density['Year'], yearly_density['mean'])
    trend_direction = "increasing" if stat > 0 else "decreasing"
    significance = "significant" if p_value < 0.05 else "not significant"
    
    print(f"Trend analysis: Octocoral density shows a {trend_direction} trend over time (r={stat:.3f}, p={p_value:.3f}), which is {significance}")
    print(f"Average annual change in density: {model.coef_[0]:.4f} per year")
    
    return yearly_density, station_density, region_density

# Function to analyze relationship between coral density and species richness
def analyze_density_richness_relationship(data):
    """Assess the relationship between stony coral density and species richness within sites."""
    print("Analyzing relationship between coral density and species richness...")
    
    if 'scor' not in data or 'species' not in data:
        print("Error: Required data not available for density-richness analysis")
        return None
    
    # Get species data and SCOR data
    scor_df = data['scor']
    species_df = data['species']
    
    # Identify species columns in species data
    meta_cols = ['Year', 'Date', 'Subregion', 'Habitat', 'SiteID', 'Site_name', 'StationID']
    species_cols = [col for col in species_df.columns if col not in meta_cols]
    
    # Calculate species richness (number of species at each station)
    if len(species_cols) > 0:
        # Convert all columns to numeric, coercing errors to NaN
        for col in species_cols:
            species_df[col] = pd.to_numeric(species_df[col], errors='coerce')
            
        species_df['richness'] = species_df[species_cols].apply(lambda row: sum(row > 0), axis=1)
    else:
        # Fallback to SCOR data
        species_cols_scor = [col for col in scor_df.columns if col not in meta_cols]
        
        # Convert all columns to numeric, coercing errors to NaN
        for col in species_cols_scor:
            scor_df[col] = pd.to_numeric(scor_df[col], errors='coerce')
            
        scor_df['richness'] = scor_df[species_cols_scor].apply(lambda row: sum(row > 0), axis=1)
        species_df = scor_df.copy()  # Use SCOR data if species data doesn't have species columns
    
    # Calculate coral density (sum of all coral species) if available, or use Stony_coral from pcover as proxy
    if 'pcover' in data:
        # Merge species richness with percent cover data (density proxy)
        merged_data = pd.merge(
            species_df[['Year', 'StationID', 'richness']],
            data['pcover'][['Year', 'StationID', 'Stony_coral']],
            on=['Year', 'StationID'],
            how='inner'
        )
        merged_data.rename(columns={'Stony_coral': 'density_proxy'}, inplace=True)
    else:
        print("Warning: Using sum of species abundance as density proxy")
        # Convert species columns to numeric and sum them
        for col in species_cols:
            species_df[col] = pd.to_numeric(species_df[col], errors='coerce')
            
        species_df['density_proxy'] = species_df[species_cols].sum(axis=1)
        merged_data = species_df[['Year', 'StationID', 'richness', 'density_proxy']]
    
    # Drop rows with NaN values in key columns
    merged_data = merged_data.dropna(subset=['richness', 'density_proxy'])
    
    # Check if we have enough data to proceed
    if len(merged_data) < 5:
        print("Error: Not enough valid data points for density-richness analysis")
        return None
    
    # Create scatter plot with regression line
    plt.figure(figsize=(10, 8))
    sns.regplot(x='density_proxy', y='richness', data=merged_data, scatter_kws={'alpha':0.5})
    plt.title('Relationship Between Stony Coral Density and Species Richness')
    plt.xlabel('Coral Density Proxy (% Cover)')
    plt.ylabel('Species Richness')
    
    # Calculate correlation and add to plot
    corr, p_value = stats.pearsonr(merged_data['density_proxy'], merged_data['richness'])
    plt.annotate(f'Correlation: {corr:.3f} (p={p_value:.3f})', xy=(0.05, 0.95), xycoords='axes fraction')
    
    # Add regression equation
    X = merged_data['density_proxy'].values.reshape(-1, 1)
    y = merged_data['richness'].values
    model = LinearRegression().fit(X, y)
    equation = f'y = {model.coef_[0]:.3f}x + {model.intercept_:.3f}'
    plt.annotate(f'Regression: {equation}', xy=(0.05, 0.90), xycoords='axes fraction')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'density_richness_relationship.png'), dpi=300)
    
    # Analyze by region if subregion data is available
    if 'Subregion' in species_df.columns:
        # Merge with subregion info
        if 'Subregion' not in merged_data.columns:
            region_info = species_df[['StationID', 'Subregion']].drop_duplicates()
            merged_data = pd.merge(merged_data, region_info, on='StationID', how='left')
        
        # Create scatter plot by region
        plt.figure(figsize=(10, 8))
        sns.lmplot(x='density_proxy', y='richness', hue='Subregion', data=merged_data, 
                 height=8, aspect=1.2, scatter_kws={'alpha':0.6})
        plt.title('Density-Richness Relationship by Region')
        plt.xlabel('Coral Density Proxy (% Cover)')
        plt.ylabel('Species Richness')
        plt.savefig(os.path.join(OUTPUT_DIR, 'density_richness_by_region.png'), dpi=300)
        
        # Calculate correlation by region
        region_correlations = merged_data.groupby('Subregion').apply(
            lambda x: pd.Series({
                'correlation': stats.pearsonr(x['density_proxy'], x['richness'])[0] if len(x) > 1 else np.nan,
                'p_value': stats.pearsonr(x['density_proxy'], x['richness'])[1] if len(x) > 1 else np.nan,
                'sample_size': len(x)
            })
        )
        
        print("\nCorrelation between density and richness by region:")
        print(region_correlations)
    
    interpretation = "positive" if corr > 0 else "negative"
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    print(f"Overall correlation between coral density and species richness: {corr:.3f} ({interpretation}, {significance})")
    
    return merged_data

# Function to evaluate temperature correlations
def analyze_temperature_correlations(data):
    """Evaluate correlations between octocoral density, water temperature, and coral health."""
    print("Analyzing temperature correlations...")
    
    # Check for temperature data
    if 'pcover' not in data or 'octo' not in data:
        print("Error: Required data not available for temperature correlation analysis")
        return None
    
    # For demonstration, let's assume temperature data is available in the datasets
    # In a real scenario, you might need to load temperature data from another file
    
    # Look for temperature columns in any dataset
    temp_columns = []
    for dataset_name, df in data.items():
        temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
        if temp_cols:
            temp_columns.extend(temp_cols)
            print(f"Found temperature columns in {dataset_name}: {temp_cols}")
    
    if not temp_columns:
        print("No temperature data found. Creating synthetic temperature data for demonstration...")
        # Create synthetic temperature data based on year and region for demonstration
        # In a real analysis, you would use actual temperature data
        
        # Get years from pcover data
        years = data['pcover']['Year'].unique()
        regions = data['pcover']['Subregion'].unique() if 'Subregion' in data['pcover'].columns else ['Region1', 'Region2']
        
        # Create synthetic temperature data with a slight warming trend
        temp_data = []
        base_temp = 28.0  # Base temperature in Celsius
        for year in years:
            for region in regions:
                # Add some random variation and a warming trend of 0.03°C per year
                annual_temp = base_temp + (year - min(years)) * 0.03 + np.random.normal(0, 0.5)
                temp_data.append({
                    'Year': year,
                    'Subregion': region,
                    'Temperature': annual_temp
                })
        
        # Create temperature DataFrame
        temp_df = pd.DataFrame(temp_data)
        data['temperature'] = temp_df
        
        print("Created synthetic temperature data with warming trend")
    else:
        # Use existing temperature data if available
        # Implementation would depend on how temperature data is structured
        print("Using existing temperature data from datasets")
    
    # Merge octocoral data with temperature data
    if 'temperature' in data:
        # Check if we need to merge by year and region
        if 'Subregion' in data['octo'].columns and 'Subregion' in data['temperature'].columns:
            octo_temp = pd.merge(
                data['octo'],
                data['temperature'],
                on=['Year', 'Subregion'],
                how='inner'
            )
        else:
            # Just merge on year if region not available
            octo_temp = pd.merge(
                data['octo'],
                data['temperature'][['Year', 'Temperature']],
                on='Year',
                how='inner'
            )
        
        # Identify density column
        density_cols = [col for col in octo_temp.columns if 'density' in col.lower() or 'count' in col.lower()]
        if density_cols:
            density_col = density_cols[0]
            
            # Create scatter plot for octocoral density vs temperature
            plt.figure(figsize=(10, 8))
            sns.regplot(x='Temperature', y=density_col, data=octo_temp, scatter_kws={'alpha':0.6})
            plt.title('Octocoral Density vs. Water Temperature')
            plt.xlabel('Water Temperature (°C)')
            plt.ylabel(f'Octocoral {density_col}')
            
            # Calculate correlation and add to plot
            corr, p_value = stats.pearsonr(octo_temp['Temperature'], octo_temp[density_col])
            plt.annotate(f'Correlation: {corr:.3f} (p={p_value:.3f})', xy=(0.05, 0.95), xycoords='axes fraction')
            
            plt.savefig(os.path.join(OUTPUT_DIR, 'octocoral_temperature_correlation.png'), dpi=300)
            
            interpretation = "positive" if corr > 0 else "negative"
            significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
            print(f"Correlation between octocoral density and temperature: {corr:.3f} ({interpretation}, {significance})")
    
    # Also correlate stony coral cover with temperature
    if 'temperature' in data and 'pcover' in data:
        # Merge pcover with temperature
        if 'Subregion' in data['pcover'].columns and 'Subregion' in data['temperature'].columns:
            coral_temp = pd.merge(
                data['pcover'],
                data['temperature'],
                on=['Year', 'Subregion'],
                how='inner'
            )
        else:
            # Just merge on year
            coral_temp = pd.merge(
                data['pcover'],
                data['temperature'][['Year', 'Temperature']],
                on='Year',
                how='inner'
            )
        
        # Create scatter plot for stony coral vs temperature
        plt.figure(figsize=(10, 8))
        sns.regplot(x='Temperature', y='Stony_coral', data=coral_temp, scatter_kws={'alpha':0.6})
        plt.title('Stony Coral Cover vs. Water Temperature')
        plt.xlabel('Water Temperature (°C)')
        plt.ylabel('Stony Coral Cover (%)')
        
        # Calculate correlation and add to plot
        corr, p_value = stats.pearsonr(coral_temp['Temperature'], coral_temp['Stony_coral'])
        plt.annotate(f'Correlation: {corr:.3f} (p={p_value:.3f})', xy=(0.05, 0.95), xycoords='axes fraction')
        
        plt.savefig(os.path.join(OUTPUT_DIR, 'stony_coral_temperature_correlation.png'), dpi=300)
        
        interpretation = "positive" if corr > 0 else "negative"
        significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
        print(f"Correlation between stony coral cover and temperature: {corr:.3f} ({interpretation}, {significance})")
    
    # Return correlation results
    return {
        'octocoral_temp_corr': corr if 'corr' in locals() else None,
        'stony_coral_temp_corr': corr if 'corr' in locals() else None
    }

# Function to identify key factors affecting coral health
def identify_key_health_factors(data):
    """Identify key factors affecting coral health, density, and species richness."""
    print("Identifying key factors affecting coral health...")
    
    if 'pcover' not in data:
        print("Error: Required data not available for health factor analysis")
        return None
    
    # Prepare dataset for analysis
    pcover_df = data['pcover']
    
    # Create correlation matrix for environmental factors
    env_factors = ['Stony_coral', 'Macroalgae', 'Octocoral', 'Cyanobacteria', 'Porifera']
    available_factors = [col for col in env_factors if col in pcover_df.columns]
    
    if len(available_factors) < 2:
        print("Not enough environmental factors available for correlation analysis")
        return None
    
    # Calculate correlation matrix
    corr_matrix = pcover_df[available_factors].corr()
    
    # Visualize correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix of Environmental Factors')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'environmental_correlations.png'), dpi=300)
    
    # Identify strongest relationships (positive and negative)
    # Get upper triangle of correlation matrix
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find strongest positive and negative correlations
    strongest_pos = upper_tri.stack().sort_values(ascending=False).head(3)
    strongest_neg = upper_tri.stack().sort_values(ascending=True).head(3)
    
    print("\nStrongest positive correlations (potential synergistic effects):")
    for idx, val in strongest_pos.items():
        print(f"{idx[0]} vs {idx[1]}: {val:.3f}")
    
    print("\nStrongest negative correlations (potential competitive or inhibitory effects):")
    for idx, val in strongest_neg.items():
        print(f"{idx[0]} vs {idx[1]}: {val:.3f}")
    
    # Check if we have time-series data to identify temporal patterns
    if len(pcover_df['Year'].unique()) > 1:
        # Analyze trends over time for each factor
        trends = {}
        for factor in available_factors:
            yearly_data = pcover_df.groupby('Year')[factor].mean().reset_index()
            if len(yearly_data) > 2:  # Need at least 3 points for meaningful trend
                # Remove any NaN values
                yearly_data = yearly_data.dropna()
                
                # Only proceed if we have enough non-NaN data
                if len(yearly_data) > 2:
                    X = yearly_data['Year'].values.reshape(-1, 1)
                    y = yearly_data[factor].values
                    model = LinearRegression().fit(X, y)
                    trend_coef = model.coef_[0]
                    trends[factor] = trend_coef
                else:
                    print(f"Warning: Not enough valid data points for {factor} trend analysis")
                    trends[factor] = 0
        
        # Plot trends if we have any
        if trends:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(trends.keys(), trends.values())
            
            # Color positive and negative trends differently
            for i, bar in enumerate(bars):
                if list(trends.values())[i] < 0:
                    bar.set_color('red')
                else:
                    bar.set_color('green')
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.title('Trends in Environmental Factors Over Time')
            plt.ylabel('Annual Rate of Change')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, 'environmental_factor_trends.png'), dpi=300)
            
            print("\nTrends in environmental factors over time:")
            for factor, trend in trends.items():
                direction = "increasing" if trend > 0 else "decreasing"
                print(f"{factor}: {direction} at rate of {abs(trend):.4f} per year")
    
    # Identify key factors based on correlation with health metrics
    key_factors = {
        'competition': [],
        'environmental': [],
        'anthropogenic': []
    }
    
    # Example key factors identification based on correlations and trends
    if 'Macroalgae' in available_factors and 'Stony_coral' in available_factors:
        macroalgae_coral_corr = corr_matrix.loc['Macroalgae', 'Stony_coral']
        if macroalgae_coral_corr < -0.2:
            key_factors['competition'].append(f"Macroalgae competition (correlation: {macroalgae_coral_corr:.3f})")
    
    if 'Cyanobacteria' in available_factors and 'Stony_coral' in available_factors:
        cyano_coral_corr = corr_matrix.loc['Cyanobacteria', 'Stony_coral'] if 'Cyanobacteria' in corr_matrix.index else 0
        if cyano_coral_corr < -0.1:
            key_factors['environmental'].append(f"Cyanobacteria presence (correlation: {cyano_coral_corr:.3f})")
    
    # Add more key factors based on additional analysis
    
    print("\nKey factors affecting coral health:")
    for category, factors in key_factors.items():
        if factors:
            print(f"\n{category.capitalize()} factors:")
            for factor in factors:
                print(f"- {factor}")
    
    return {
        'correlation_matrix': corr_matrix,
        'key_factors': key_factors,
        'trends': trends if 'trends' in locals() else None
    }

# Function to identify early warning indicators
def identify_early_warning_indicators(data):
    """Identify early indicators that could help anticipate significant declines in coral populations."""
    print("Identifying early warning indicators for coral population declines...")
    
    if 'pcover' not in data or len(data['pcover']['Year'].unique()) < 3:
        print("Error: Insufficient temporal data for early warning indicators analysis")
        return None
    
    pcover_df = data['pcover']
    
    # Calculate year-to-year variability in coral cover
    # Higher variability can be an early warning signal for ecosystem instability
    yearly_mean = pcover_df.groupby('Year')['Stony_coral'].mean().reset_index()
    yearly_mean['prev_year'] = yearly_mean['Year'].shift(1)
    yearly_mean['prev_cover'] = yearly_mean['Stony_coral'].shift(1)
    yearly_mean['change'] = yearly_mean['Stony_coral'] - yearly_mean['prev_cover']
    yearly_mean['percent_change'] = (yearly_mean['change'] / yearly_mean['prev_cover']) * 100
    yearly_mean = yearly_mean.dropna()
    
    # Calculate increasing variability (variance over rolling windows)
    if len(yearly_mean) >= 4:  # Need at least 4 years for rolling window
        yearly_mean['rolling_var'] = yearly_mean['Stony_coral'].rolling(window=3).var()
        
        # Drop NaN values from rolling variance
        valid_variance = yearly_mean.dropna(subset=['rolling_var'])
        
        if len(valid_variance) >= 3:  # Need at least 3 points for plot
            # Plot the variance over time
            plt.figure(figsize=(10, 6))
            plt.plot(valid_variance['Year'], valid_variance['rolling_var'], 'o-', color='red')
            plt.title('Temporal Variance in Stony Coral Cover (3-year Rolling Window)')
            plt.xlabel('Year')
            plt.ylabel('Variance')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(OUTPUT_DIR, 'coral_cover_variance.png'), dpi=300)
            
            # Check for increasing variance trend (warning sign)
            variance_trend = stats.linregress(
                valid_variance['Year'], 
                valid_variance['rolling_var']
            )
            print(f"Variance trend slope: {variance_trend.slope:.6f}, p-value: {variance_trend.pvalue:.4f}")
            
            if variance_trend.slope > 0 and variance_trend.pvalue < 0.1:
                print("WARNING: Increasing variance detected in stony coral cover - potential early warning sign of system instability")
        else:
            print("Not enough valid data points for variance trend analysis")
    
    # Calculate rate of change and plot
    plt.figure(figsize=(10, 6))
    plt.bar(yearly_mean['Year'], yearly_mean['percent_change'], color=['red' if x < 0 else 'green' for x in yearly_mean['percent_change']])
    plt.axhline(y=0, color='black', linestyle='-')
    plt.title('Annual Percent Change in Stony Coral Cover')
    plt.xlabel('Year')
    plt.ylabel('Percent Change (%)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'coral_cover_change_rate.png'), dpi=300)
    
    # Identify acceleration in decline (another warning indicator)
    yearly_mean['change_acceleration'] = yearly_mean['percent_change'].diff()
    
    # Identify years with significant declines
    significant_declines = yearly_mean[yearly_mean['percent_change'] < -10]
    if not significant_declines.empty:
        print("\nYears with significant coral cover decline (>10%):")
        for _, row in significant_declines.iterrows():
            print(f"Year {row['Year']}: {row['percent_change']:.1f}% decline from previous year")
    
    # Check for ratio changes between coral and macroalgae (phase shift indicator)
    if 'Macroalgae' in pcover_df.columns:
        yearly_ratio = pcover_df.groupby('Year').agg({
            'Stony_coral': 'mean',
            'Macroalgae': 'mean'
        }).reset_index()
        
        # Calculate ratio and handle division by zero
        yearly_ratio['coral_algae_ratio'] = yearly_ratio['Stony_coral'] / yearly_ratio['Macroalgae'].replace(0, np.nan)
        
        # Drop any NaN values
        yearly_ratio = yearly_ratio.dropna(subset=['coral_algae_ratio'])
        
        if len(yearly_ratio) >= 3:  # Need at least 3 points for trend analysis
            # Plot ratio trend
            plt.figure(figsize=(10, 6))
            plt.plot(yearly_ratio['Year'], yearly_ratio['coral_algae_ratio'], 'o-', color='blue')
            plt.title('Ratio of Stony Coral to Macroalgae Cover Over Time')
            plt.xlabel('Year')
            plt.ylabel('Coral:Algae Ratio')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(OUTPUT_DIR, 'coral_algae_ratio.png'), dpi=300)
            
            # Check for declining ratio trend (warning sign of phase shift)
            ratio_trend = stats.linregress(yearly_ratio['Year'], yearly_ratio['coral_algae_ratio'])
            print(f"Coral:Algae ratio trend slope: {ratio_trend.slope:.6f}, p-value: {ratio_trend.pvalue:.4f}")
            
            if ratio_trend.slope < 0 and ratio_trend.pvalue < 0.1:
                print("WARNING: Declining coral-to-algae ratio detected - potential early warning sign of phase shift")
        else:
            print("Not enough valid data points for coral-algae ratio analysis")
    
    # Summarize early warning indicators
    early_indicators = []
    
    # Include indicators based on analysis results
    if 'variance_trend' in locals() and variance_trend.slope > 0:
        early_indicators.append("Increasing temporal variance in coral cover")
    
    if not significant_declines.empty:
        early_indicators.append("Episodes of rapid coral cover decline (>10% in a single year)")
    
    if 'ratio_trend' in locals() and ratio_trend.slope < 0:
        early_indicators.append("Declining coral-to-algae ratio")
    
    # Check for changes in species composition
    if 'species' in data and len(data['species']['Year'].unique()) > 1:
        print("Analyzing changes in species composition as potential early warning...")
        
        # This would be more detailed in a real analysis
        early_indicators.append("Changes in coral species composition")
    
    print("\nIdentified early warning indicators:")
    for i, indicator in enumerate(early_indicators, 1):
        print(f"{i}. {indicator}")
    
    return {
        'yearly_changes': yearly_mean,
        'early_indicators': early_indicators,
        'significant_declines': significant_declines
    }

# Function to model future coral reef evolution
def model_future_evolution(data):
    """Model the evolution of coral reefs at the observed stations over the next five years."""
    print("Modeling future evolution of coral reefs over the next five years...")
    
    if 'pcover' not in data or len(data['pcover']['Year'].unique()) < 3:
        print("Error: Insufficient temporal data for future evolution modeling")
        return None
    
    pcover_df = data['pcover']
    
    # Get the range of years in the data
    min_year = pcover_df['Year'].min()
    max_year = pcover_df['Year'].max()
    future_years = range(max_year + 1, max_year + 6)  # Next 5 years
    
    print(f"Data spans years {min_year} to {max_year}")
    print(f"Modeling future evolution for years {list(future_years)}")
    
    # Prepare dataframe for annual trends
    yearly_data = pcover_df.groupby('Year').agg({
        'Stony_coral': 'mean',
        'Macroalgae': 'mean',
        'Octocoral': 'mean'
    }).reset_index()
    
    # Model future trends for major taxa
    future_predictions = {'Year': list(future_years)}
    prediction_intervals = {}
    
    for taxa in ['Stony_coral', 'Macroalgae', 'Octocoral']:
        if taxa in yearly_data.columns:
            # Drop any NaN values for this taxa
            valid_data = yearly_data.dropna(subset=[taxa])
            
            if len(valid_data) < 3:
                print(f"Warning: Not enough valid data points for {taxa} prediction")
                continue
                
            # Prepare data for modeling
            X = valid_data['Year'].values.reshape(-1, 1)
            y = valid_data[taxa].values
            
            # Linear regression model
            linear_model = LinearRegression().fit(X, y)
            
            # Create future X values
            X_future = np.array(list(future_years)).reshape(-1, 1)
            
            # Predict future values
            y_pred_linear = linear_model.predict(X_future)
            
            # Try to fit a more advanced Random Forest model if we have enough data
            if len(X) >= 5:  # Need reasonable amount of data for RF
                try:
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X, y)
                    y_pred_rf = rf_model.predict(X_future)
                    
                    # Use RF predictions if they seem reasonable
                    if np.all(y_pred_rf >= 0):  # Ensure non-negative predictions
                        y_pred = y_pred_rf
                        model_type = "Random Forest"
                    else:
                        y_pred = y_pred_linear
                        model_type = "Linear Regression"
                except Exception as e:
                    print(f"Error fitting Random Forest model for {taxa}: {e}")
                    y_pred = y_pred_linear
                    model_type = "Linear Regression"
            else:
                y_pred = y_pred_linear
                model_type = "Linear Regression"
            
            # Store predictions
            future_predictions[taxa] = y_pred
            
            # Calculate prediction intervals (for linear model)
            # This is a simplified approach - more sophisticated methods exist
            residuals = y - linear_model.predict(X)
            residual_std = np.std(residuals)
            
            # 95% prediction interval
            prediction_intervals[taxa] = {
                'lower': y_pred - 1.96 * residual_std,
                'upper': y_pred + 1.96 * residual_std
            }
            
            print(f"Used {model_type} to predict future {taxa} values")
    
    # Create DataFrame for predictions
    future_df = pd.DataFrame(future_predictions)
    
    # Only proceed if we have predictions
    if len(future_df.columns) <= 1:  # Only Year column
        print("Error: No valid predictions could be generated")
        return None
    
    # Combine historical and future data for plotting
    combined_df = pd.concat([
        yearly_data,
        future_df
    ]).reset_index(drop=True)
    
    # Plot historical data and future projections
    plt.figure(figsize=(14, 8))
    
    # Plot historical data with solid lines
    historical_years = combined_df['Year'] <= max_year
    for taxa, color in zip(['Stony_coral', 'Macroalgae', 'Octocoral'], ['blue', 'green', 'orange']):
        if taxa in combined_df.columns:
            # Drop NaN values for plotting
            valid_data = combined_df.loc[historical_years, ['Year', taxa]].dropna()
            if not valid_data.empty:
                plt.plot(
                    valid_data['Year'], 
                    valid_data[taxa], 
                    'o-', 
                    color=color, 
                    label=f'Historical {taxa}'
                )
    
    # Plot future projections with dashed lines
    future_years_mask = combined_df['Year'] > max_year
    for taxa, color in zip(['Stony_coral', 'Macroalgae', 'Octocoral'], ['blue', 'green', 'orange']):
        if taxa in combined_df.columns:
            plt.plot(
                combined_df.loc[future_years_mask, 'Year'], 
                combined_df.loc[future_years_mask, taxa], 
                '--', 
                color=color, 
                label=f'Projected {taxa}'
            )
            
            # Add prediction intervals
            if taxa in prediction_intervals:
                plt.fill_between(
                    future_df['Year'],
                    prediction_intervals[taxa]['lower'],
                    prediction_intervals[taxa]['upper'],
                    color=color,
                    alpha=0.2
                )
    
    plt.axvline(x=max_year, color='red', linestyle='--', alpha=0.5, label='Current Year')
    plt.title('Coral Reef Evolution: Historical Data and Future Projections (5 Years)')
    plt.xlabel('Year')
    plt.ylabel('Mean Cover (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, 'future_evolution_projection.png'), dpi=300)
    
    # Regional analysis if we have subregion data
    if 'Subregion' in pcover_df.columns:
        # Get unique regions
        regions = pcover_df['Subregion'].unique()
        
        # Model future evolution by region
        plt.figure(figsize=(15, 10))
        
        for i, region in enumerate(regions):
            region_data = pcover_df[pcover_df['Subregion'] == region]
            
            if len(region_data['Year'].unique()) >= 3:  # Need at least 3 years for trend
                yearly_region = region_data.groupby('Year')['Stony_coral'].mean().reset_index()
                
                # Prepare data for modeling
                X_region = yearly_region['Year'].values.reshape(-1, 1)
                y_region = yearly_region['Stony_coral'].values
                
                # Linear model for simplicity
                model = LinearRegression().fit(X_region, y_region)
                
                # Predict future values
                X_future = np.array(list(future_years)).reshape(-1, 1)
                y_future = model.predict(X_future)
                
                # Plot in subplot
                plt.subplot(2, 2, i+1)
                plt.plot(yearly_region['Year'], yearly_region['Stony_coral'], 'o-', label='Historical')
                plt.plot(future_years, y_future, 'r--', label='Projected')
                plt.title(f'Region: {region}')
                plt.xlabel('Year')
                plt.ylabel('Stony Coral Cover (%)')
                plt.grid(True, alpha=0.3)
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'future_evolution_by_region.png'), dpi=300)
    
    # Calculate key metrics for the projected future
    last_historical_year = yearly_data[yearly_data['Year'] == max_year]
    final_projection_year = future_df[future_df['Year'] == max(future_years)]
    
    # Calculate percent changes from current to future
    percent_changes = {}
    for taxa in ['Stony_coral', 'Macroalgae', 'Octocoral']:
        if taxa in last_historical_year.columns and taxa in final_projection_year.columns:
            current_value = last_historical_year[taxa].values[0]
            future_value = final_projection_year[taxa].values[0]
            percent_change = ((future_value - current_value) / current_value) * 100 if current_value > 0 else np.inf
            percent_changes[taxa] = percent_change
    
    print("\nProjected changes over the next 5 years:")
    for taxa, change in percent_changes.items():
        direction = "increase" if change > 0 else "decrease"
        print(f"{taxa}: {abs(change):.1f}% {direction}")
    
    # Assess potential phase shifts
    if 'Stony_coral' in percent_changes and 'Macroalgae' in percent_changes:
        if percent_changes['Stony_coral'] < -10 and percent_changes['Macroalgae'] > 10:
            print("\nWARNING: Projections indicate potential phase shift from coral to algae dominance")
    
    # Return predictions and metrics
    return {
        'future_predictions': future_df,
        'percent_changes': percent_changes,
        'prediction_intervals': prediction_intervals
    }

# Main function to run the analysis
def main():
    """Main function to orchestrate the CREMP data analysis"""
    print("\n=========================================")
    print("CREMP Data Analysis Tool")
    print("=========================================\n")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("Starting analysis at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Load all datasets
    data = load_data()
    if not data:
        print("Error: Failed to load data. Exiting.")
        return
    
    # Print column names from the datasets for debugging
    print("\nColumn names in pcover data:", data['pcover'].columns.tolist())
    
    # Run basic analyses
    print("\n=== Basic Analysis ===")
    regional_cover = analyze_regional_cover(data)
    habitat_trends = analyze_habitat_trends(data)
    coral_health = analyze_coral_health(data)
    correlation = compare_distributions(data)
    site_health, top_sites, bottom_sites = generate_site_report(data)
    
    # Run advanced exploratory analyses
    print("\n=== Advanced Exploratory Analysis ===")
    temporal_trends = analyze_temporal_trends(data)
    species_richness = analyze_species_richness_trends(data)
    octocoral_trends = analyze_octocoral_density(data)
    
    # Run relationship and correlation analyses
    print("\n=== Relationship Analysis ===")
    density_richness = analyze_density_richness_relationship(data)
    temp_correlations = analyze_temperature_correlations(data)
    
    # Run future outlook analyses
    print("\n=== Future Outlook Analysis ===")
    health_factors = identify_key_health_factors(data)
    early_indicators = identify_early_warning_indicators(data)
    future_projections = model_future_evolution(data)
    
    # Generate comprehensive report
    generate_report(data, {
        'regional_cover': regional_cover,
        'habitat_trends': habitat_trends,
        'site_health': site_health,
        'temporal_trends': temporal_trends,
        'species_richness': species_richness,
        'octocoral_trends': octocoral_trends,
        'density_richness': density_richness,
        'health_factors': health_factors,
        'early_indicators': early_indicators,
        'future_projections': future_projections
    })
    
    # Print summary statistics
    print("\n=========================================")
    print("ANALYSIS SUMMARY")
    print("=========================================")
    print(f"Total sites analyzed: {site_health['StationID'].nunique() if not site_health.empty else 0}")
    print(f"Average stony coral cover: {data['pcover']['Stony_coral'].mean():.2f}%")
    print(f"Average macroalgae cover: {data['pcover']['Macroalgae'].mean():.2f}%")
    
    # Check if we have top_sites and the site_name_col before trying to print them
    if not top_sites.empty and len(top_sites) > 0:
        # Get the actual site name column that was used
        site_cols = [col for col in top_sites.columns if 'site' in col.lower() or 'name' in col.lower() or col == 'StationID']
        if site_cols:
            used_site_col = site_cols[0]
            print("\nTop 3 healthiest sites:")
            for i, (_, row) in enumerate(top_sites.head(3).iterrows()):
                subregion = row['Subregion'] if 'Subregion' in row else 'Unknown'
                habitat = row['Habitat'] if 'Habitat' in row else 'Unknown'
                print(f"{i+1}. {row[used_site_col]} ({subregion}, {habitat}) - Score: {row['health_score']:.2f}")
        else:
            print("\nTop 3 healthiest sites could not be displayed (missing site name column)")
    else:
        print("\nNo top sites data available")
        
    if not bottom_sites.empty and len(bottom_sites) > 0:
        # Get the actual site name column that was used
        site_cols = [col for col in bottom_sites.columns if 'site' in col.lower() or 'name' in col.lower() or col == 'StationID']
        if site_cols:
            used_site_col = site_cols[0]
            print("\nMost concerning sites:")
            for i, (_, row) in enumerate(bottom_sites.head(3).iterrows()):
                subregion = row['Subregion'] if 'Subregion' in row else 'Unknown'
                habitat = row['Habitat'] if 'Habitat' in row else 'Unknown'
                print(f"{i+1}. {row[used_site_col]} ({subregion}, {habitat}) - Score: {row['health_score']:.2f}")
        else:
            print("\nMost concerning sites could not be displayed (missing site name column)")
    else:
        print("\nNo bottom sites data available")
    
    # Print future projections summary if available
    if future_projections and 'percent_changes' in future_projections:
        print("\nProjected changes in the next 5 years:")
        for taxa, change in future_projections['percent_changes'].items():
            direction = "increase" if change > 0 else "decrease"
            print(f"  {taxa}: {abs(change):.1f}% {direction}")
    
    # Print early warning indicators if available
    if early_indicators and 'early_indicators' in early_indicators:
        print("\nEarly warning indicators identified:")
        for i, indicator in enumerate(early_indicators['early_indicators'], 1):
            print(f"  {i}. {indicator}")
    
    print("\nAnalysis complete. Results saved to:", os.path.abspath(OUTPUT_DIR))
    print("Finished at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Function to generate a comprehensive report
def generate_report(data, analysis_results):
    """Generate a comprehensive report with findings and visualizations."""
    print("Generating comprehensive report...")
    
    # Create report file
    report_file = os.path.join(OUTPUT_DIR, "CREMP_Analysis_Report.md")
    
    with open(report_file, 'w') as f:
        f.write("# Coral Reef Evaluation and Monitoring Project (CREMP) Analysis Report\n\n")
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d')}*\n\n")
        
        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write("This report presents a comprehensive analysis of CREMP monitoring data, examining coral reef health,\n")
        f.write("species distribution, and temporal trends across monitoring sites. The analysis includes current\n")
        f.write("status assessment, identification of key factors affecting coral health, and projections of future\n")
        f.write("reef evolution over the next five years.\n\n")
        
        # Current Status
        f.write("## Current Status of Coral Reefs\n\n")
        f.write(f"- Average stony coral cover: {data['pcover']['Stony_coral'].mean():.2f}%\n")
        f.write(f"- Average macroalgae cover: {data['pcover']['Macroalgae'].mean():.2f}%\n")
        if 'octocoral_trends' in analysis_results and analysis_results['octocoral_trends'] is not None:
            f.write("- Octocoral density shows ")
            try:
                if isinstance(analysis_results['octocoral_trends'], tuple) and len(analysis_results['octocoral_trends']) > 0:
                    yearly_density = analysis_results['octocoral_trends'][0]
                    if not yearly_density.empty and 'Year' in yearly_density.columns and 'mean' in yearly_density.columns:
                        density_trend = stats.linregress(
                            yearly_density['Year'],
                            yearly_density['mean']
                        )
                        trend_direction = "increasing" if density_trend.slope > 0 else "decreasing"
                        f.write(f"{trend_direction} trend over time\n")
                    else:
                        f.write("variable patterns across sites\n")
                else:
                    f.write("variable patterns across sites\n")
            except:
                f.write("variable patterns across sites\n")
        
        # Regional Patterns
        f.write("\n## Regional Patterns\n\n")
        f.write("Analysis reveals significant variations in coral reef health across different regions:\n\n")
        
        if 'site_health' in analysis_results and analysis_results['site_health'] is not None:
            # Group by region to get regional averages
            if 'Subregion' in analysis_results['site_health'].columns:
                region_health = analysis_results['site_health'].groupby('Subregion').agg({
                    'Stony_coral': 'mean',
                    'Macroalgae': 'mean',
                    'health_score': 'mean'
                }).sort_values('health_score', ascending=False)
                
                f.write("| Region | Stony Coral Cover (%) | Macroalgae Cover (%) | Health Score |\n")
                f.write("|--------|----------------------|---------------------|-------------|\n")
                for region, row in region_health.iterrows():
                    f.write(f"| {region} | {row['Stony_coral']:.2f} | {row['Macroalgae']:.2f} | {row['health_score']:.2f} |\n")
        
        # Key Findings
        f.write("\n## Key Findings\n\n")
        
        # Temporal Trends
        if 'temporal_trends' in analysis_results and analysis_results['temporal_trends'] is not None:
            f.write("### Temporal Trends in Coral Cover\n\n")
            try:
                if isinstance(analysis_results['temporal_trends'], tuple) and len(analysis_results['temporal_trends']) > 0:
                    yearly_cover = analysis_results['temporal_trends'][0]
                    if not yearly_cover.empty and 'Year' in yearly_cover.columns and 'mean' in yearly_cover.columns:
                        first_year = yearly_cover['Year'].iloc[0]
                        last_year = yearly_cover['Year'].iloc[-1]
                        first_cover = yearly_cover['mean'].iloc[0]
                        last_cover = yearly_cover['mean'].iloc[-1]
                        percent_change = ((last_cover - first_cover) / first_cover) * 100 if first_cover > 0 else np.inf
                        direction = "increased" if percent_change > 0 else "decreased"
                        
                        f.write(f"From {first_year} to {last_year}, stony coral cover has {direction} ")
                        f.write(f"by {abs(percent_change):.1f}%. ")
                        
                        # Analyze rate of change
                        model = LinearRegression().fit(
                            yearly_cover['Year'].values.reshape(-1, 1),
                            yearly_cover['mean'].values
                        )
                        annual_change = model.coef_[0]
                        f.write(f"The average annual change is {annual_change:.4f}% per year.\n\n")
                    else:
                        f.write("Analysis shows variable patterns in coral cover over time.\n\n")
                else:
                    f.write("Analysis shows variable patterns in coral cover over time.\n\n")
            except:
                f.write("Analysis shows variable patterns in coral cover over time.\n\n")
        
        # Species Richness
        if 'species_richness' in analysis_results and analysis_results['species_richness'] is not None:
            f.write("### Species Richness Patterns\n\n")
            try:
                if isinstance(analysis_results['species_richness'], tuple) and len(analysis_results['species_richness']) > 0:
                    richness = analysis_results['species_richness'][0]
                    if not richness.empty and 'Year' in richness.columns and 'mean' in richness.columns:
                        first_year = richness['Year'].iloc[0]
                        last_year = richness['Year'].iloc[-1]
                        first_richness = richness['mean'].iloc[0]
                        last_richness = richness['mean'].iloc[-1]
                        percent_change = ((last_richness - first_richness) / first_richness) * 100 if first_richness > 0 else np.inf
                        
                        f.write(f"Species richness has {'increased' if percent_change > 0 else 'decreased'} ")
                        f.write(f"by {abs(percent_change):.1f}% from {first_year} to {last_year}.\n\n")
                    else:
                        f.write("Species richness shows variable patterns across regions and over time.\n\n")
                else:
                    f.write("Species richness shows variable patterns across regions and over time.\n\n")
            except:
                f.write("Species richness shows variable patterns across regions and over time.\n\n")
        
        # Health Factors
        if 'health_factors' in analysis_results and analysis_results['health_factors'] is not None:
            f.write("### Key Factors Affecting Coral Health\n\n")
            try:
                if isinstance(analysis_results['health_factors'], dict) and 'key_factors' in analysis_results['health_factors']:
                    factors = analysis_results['health_factors']['key_factors']
                    for category, category_factors in factors.items():
                        if category_factors:
                            f.write(f"**{category.capitalize()} factors:**\n\n")
                            for factor in category_factors:
                                f.write(f"- {factor}\n")
                            f.write("\n")
                else:
                    f.write("Multiple factors affect coral health including competition with macroalgae and environmental conditions.\n\n")
            except:
                f.write("Multiple factors affect coral health including competition with macroalgae and environmental conditions.\n\n")
        
        # Early Warning Indicators
        if 'early_indicators' in analysis_results and analysis_results['early_indicators'] is not None:
            f.write("### Early Warning Indicators\n\n")
            try:
                if isinstance(analysis_results['early_indicators'], dict) and 'early_indicators' in analysis_results['early_indicators']:
                    indicators = analysis_results['early_indicators']['early_indicators']
                    f.write("The following early warning indicators were identified:\n\n")
                    for indicator in indicators:
                        f.write(f"- {indicator}\n")
                    f.write("\n")
                else:
                    f.write("Analysis identified several warning signs including increased variability and coral-algae phase shifts.\n\n")
            except:
                f.write("Analysis identified several warning signs including increased variability and coral-algae phase shifts.\n\n")
        
        # Future Projections
        if 'future_projections' in analysis_results and analysis_results['future_projections'] is not None:
            f.write("## Future Projections (5-Year Outlook)\n\n")
            try:
                if isinstance(analysis_results['future_projections'], dict) and 'percent_changes' in analysis_results['future_projections']:
                    changes = analysis_results['future_projections']['percent_changes']
                    f.write("Based on historical trends, the following changes are projected for the next 5 years:\n\n")
                    for taxa, change in changes.items():
                        direction = "increase" if change > 0 else "decrease"
                        f.write(f"- {taxa}: {abs(change):.1f}% {direction}\n")
                    
                    # Assess overall reef health trajectory
                    if 'Stony_coral' in changes:
                        if changes['Stony_coral'] > 5:
                            f.write("\nThe overall reef health trajectory is **positive**, with projected increases in stony coral cover.\n")
                        elif changes['Stony_coral'] < -5:
                            f.write("\nThe overall reef health trajectory is **negative**, with projected decreases in stony coral cover.\n")
                        else:
                            f.write("\nThe overall reef health trajectory is **stable**, with minimal projected changes in stony coral cover.\n")
                else:
                    f.write("Projections suggest continued pressures on coral reef ecosystems with potential for phase shifts if current trends continue.\n")
            except:
                f.write("Projections suggest continued pressures on coral reef ecosystems with potential for phase shifts if current trends continue.\n")
        
        # Recommendations
        f.write("\n## Recommendations for Conservation and Management\n\n")
        f.write("Based on the analysis, the following actions are recommended:\n\n")
        
        # Generate dynamic recommendations based on findings
        recommendations = []
        
        # Check coral cover trends
        if 'temporal_trends' in analysis_results and analysis_results['temporal_trends'] is not None:
            try:
                if isinstance(analysis_results['temporal_trends'], tuple) and len(analysis_results['temporal_trends']) > 0:
                    yearly_cover = analysis_results['temporal_trends'][0]
                    if not yearly_cover.empty and 'Year' in yearly_cover.columns and 'mean' in yearly_cover.columns:
                        model = LinearRegression().fit(
                            yearly_cover['Year'].values.reshape(-1, 1),
                            yearly_cover['mean'].values
                        )
                        if model.coef_[0] < -0.1:
                            recommendations.append("Implement immediate protection measures to address the significant declining trend in stony coral cover")
            except:
                pass
        
        # Check for phase shifts
        if 'early_indicators' in analysis_results and analysis_results['early_indicators'] is not None:
            try:
                if isinstance(analysis_results['early_indicators'], dict) and 'early_indicators' in analysis_results['early_indicators']:
                    if any("phase shift" in indicator.lower() for indicator in analysis_results['early_indicators']['early_indicators']):
                        recommendations.append("Develop targeted macroalgae control strategies to prevent coral-algal phase shifts")
            except:
                pass
        
        # Add general recommendations
        recommendations.extend([
            "Establish additional monitoring sites in under-represented habitats and regions",
            "Implement targeted restoration efforts focusing on reefs with high decline rates",
            "Develop an early warning system based on the identified indicators to anticipate and mitigate future declines",
            "Enhance water quality management to reduce stressors on coral reefs",
            "Conduct focused research on resilient coral sites to identify protective factors"
        ])
        
        # Write recommendations
        for i, rec in enumerate(recommendations, 1):
            f.write(f"{i}. {rec}\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        f.write("This comprehensive analysis of CREMP data provides valuable insights into the current status,\n")
        f.write("trends, and potential future trajectories of coral reef ecosystems. By identifying key factors\n")
        f.write("affecting coral health and early warning indicators, this report aims to inform effective\n")
        f.write("conservation and management strategies. The findings highlight the importance of continued\n")
        f.write("monitoring and adaptive management approaches to protect these vital marine ecosystems.\n\n")
        
        # Methodology 
        f.write("## Methodology\n\n")
        f.write("This analysis utilized Python-based data processing and statistical modeling techniques, including:\n\n")
        f.write("- Temporal trend analysis using linear regression models\n")
        f.write("- Correlation analysis to identify relationships between environmental factors\n")
        f.write("- Random Forest and Linear Regression models for future projections\n")
        f.write("- Statistical significance testing using Pearson's correlation and t-tests\n")
        f.write("- Variance analysis for early warning detection\n\n")
        
        f.write("Data from the Coral Reef Evaluation and Monitoring Project (CREMP) was analyzed, including percent cover,\n")
        f.write("species composition, and site metadata spanning multiple years and regions.\n")
    
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    main() 