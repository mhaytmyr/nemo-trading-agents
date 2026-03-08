"""
Register climate analysis tools for NAT.
This wraps our standalone Python functions as NAT tools.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

# # Import our standalone tools
# from src.climate_tools_simple import (
#     load_climate_data,
#     calculate_statistics,
#     filter_by_country,
#     find_extreme_years,
#     create_visualization,
#     list_countries
# )



def load_climate_data(file_path: str = "temperature_annual.csv") -> pd.DataFrame:
    """
    Load climate temperature data from CSV file.

    Args:
        file_path: Path to the CSV file (monthly or annual)

    Returns:
        DataFrame with temperature data
    """
    df = pd.read_csv(file_path)
    return df


def calculate_statistics(df: pd.DataFrame, country: Optional[str] = None) -> str:
    """
    Calculate basic statistics from temperature data.

    Args:
        df: DataFrame with temperature data
        country: Optional country name to filter by

    Returns:
        JSON string with statistics
    """
    # Filter by country if specified
    if country and 'country_name' in df.columns:
        df = df[df['country_name'] == country]
        if df.empty:
            return json.dumps({"error": f"No data found for country: {country}"})

    # Determine temperature column
    temp_col = 'annual_temperature' if 'annual_temperature' in df.columns else 'temperature'

    stats = {
        "mean_temperature": round(float(df[temp_col].mean()), 2),
        "min_temperature": round(float(df[temp_col].min()), 2),
        "max_temperature": round(float(df[temp_col].max()), 2),
        "std_deviation": round(float(df[temp_col].std()), 2),
        "num_records": len(df)
    }

    # Calculate trend if we have yearly data
    if 'year' in df.columns and 'annual_temperature' in df.columns:
        yearly_global = df.groupby('year')['annual_temperature'].mean()
        if len(yearly_global) > 1:
            years = yearly_global.index.values
            temps = yearly_global.values
            z = np.polyfit(years, temps, 1)
            stats['trend_per_decade'] = round(float(z[0] * 10), 3)
            stats['years_analyzed'] = f"{years.min()}-{years.max()}"

    if country:
        stats['country'] = country

    return json.dumps(stats, indent=2)


def filter_by_country(df: pd.DataFrame, country_name: str) -> pd.DataFrame:
    """
    Filter temperature data by country name.

    Args:
        df: Temperature data DataFrame
        country_name: Country name (e.g., 'United States', 'France')

    Returns:
        Filtered DataFrame as JSON string
    """
    filtered = df[df['country_name'] == country_name]

    if filtered.empty:
        return json.dumps({"error": f"No data found for country: {country_name}"})

    # Return summary info
    result = {
        "country": country_name,
        "stations": filtered['station_id'].nunique(),
        "records": len(filtered),
        "years": f"{filtered['year'].min()}-{filtered['year'].max()}",
        "stations_list": filtered[['station_id', 'name']].drop_duplicates().to_dict('records')
    }

    return json.dumps(result, indent=2)


def create_visualization(df: pd.DataFrame,
                         plot_type: str = "annual_trend",
                         country: Optional[str] = None,
                         save_path: str = "climate_plot.png") -> str:
    """
    Create climate data visualizations and save to file.

    Args:
        df: Temperature data DataFrame
        plot_type: Type of plot ('annual_trend', 'country_comparison', 'monthly_pattern')
        country: Optional country to focus on
        save_path: Path to save the plot

    Returns:
        Description of what was plotted
    """
    plt.figure(figsize=(10, 6))

    # Filter by country if specified
    if country and 'country_name' in df.columns:
        df = df[df['country_name'] == country]
        if df.empty:
            return f"No data found for country: {country}"

    if plot_type == "annual_trend":
        # Calculate global annual means
        if 'annual_temperature' in df.columns:
            annual_means = df.groupby('year')['annual_temperature'].mean()
        else:
            annual_means = df.groupby('year')['temperature'].mean()

        plt.plot(annual_means.index, annual_means.values, 'b-', linewidth=2)
        plt.scatter(annual_means.index, annual_means.values, alpha=0.6)

        # Add trend line
        z = np.polyfit(annual_means.index, annual_means.values, 1)
        p = np.poly1d(z)
        plt.plot(annual_means.index, p(annual_means.index), "r--", alpha=0.8,
                 label=f'Trend: {z[0] * 10:.3f}°C/decade')

        plt.xlabel('Year')
        plt.ylabel('Temperature (°C)')
        title = f'Annual Average Temperature Trend'
        if country:
            title += f' - {country}'
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)

    elif plot_type == "country_comparison" and 'country_name' in df.columns:
        # Compare top 5 countries by temperature change
        country_trends = {}
        for country_name in df['country_name'].unique():
            country_data = df[df['country_name'] == country_name]
            if 'annual_temperature' in df.columns:
                yearly = country_data.groupby('year')['annual_temperature'].mean()
            else:
                yearly = country_data.groupby('year')['temperature'].mean()

            if len(yearly) > 10:  # Need enough data for trend
                z = np.polyfit(yearly.index, yearly.values, 1)
                country_trends[country_name] = z[0] * 10  # Per decade

        # Sort by trend and plot top 5
        sorted_countries = sorted(country_trends.items(), key=lambda x: x[1], reverse=True)[:5]

        countries = [c[0] for c in sorted_countries]
        trends = [c[1] for c in sorted_countries]

        plt.bar(countries, trends, color='coral', edgecolor='darkred')
        plt.xlabel('Country')
        plt.ylabel('Temperature Trend (°C/decade)')
        plt.title('Top 5 Countries by Warming Trend')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')

    elif plot_type == "monthly_pattern" and 'month' in df.columns:
        # Monthly temperature pattern
        monthly_means = df.groupby('month')['temperature'].mean()

        plt.bar(monthly_means.index, monthly_means.values, color='skyblue', edgecolor='navy')
        plt.xlabel('Month')
        plt.ylabel('Average Temperature (°C)')
        title = 'Monthly Temperature Pattern'
        if country:
            title += f' - {country}'
        plt.title(title)
        plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        plt.grid(True, alpha=0.3)

    else:
        plt.close()
        return f"Plot type '{plot_type}' not available for this data"

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return f"Created {plot_type} plot and saved to {save_path}"


def list_countries(df: pd.DataFrame) -> str:
    """
    List all available countries in the dataset.

    Args:
        df: Temperature data DataFrame

    Returns:
        JSON string with list of countries
    """
    countries = sorted(df['country_name'].unique())

    return json.dumps({
        "available_countries": countries,
        "total_count": len(countries)
    }, indent=2)


def find_extreme_years(df: pd.DataFrame, n: int = 5, extreme_type: str = "warmest") -> str:
    """
    Find the warmest or coldest years globally.

    Args:
        df: Temperature data DataFrame
        n: Number of years to return
        extreme_type: 'warmest' or 'coldest'

    Returns:
        JSON string with extreme years
    """
    # Calculate global annual means
    if 'annual_temperature' in df.columns:
        annual_means = df.groupby('year')['annual_temperature'].mean()
    else:
        annual_means = df.groupby('year')['temperature'].mean()

    # Sort appropriately
    sorted_years = annual_means.sort_values(ascending=(extreme_type == 'coldest'))

    results = []
    for i, (year, temp) in enumerate(sorted_years.head(n).items()):
        results.append({
            "rank": i + 1,
            "year": int(year),
            "temperature": round(float(temp), 2)
        })

    return json.dumps({
        "type": extreme_type,
        "years": results
    }, indent=2)

# Base path to climate data - use absolute path
DATA_PATH = os.path.join("/home/admin1/Desktop/nemo-trading-agents", "resources", "climate_data", "temperature_annual.csv")


# 1. Input schemas tell LLM what each tool expects
class CalculateStatsInput(BaseModel):
    country: str = Field(
        default="",
        description="Country name to filter by (e.g., 'United States', 'France'). Leave empty for global statistics."
    )


class FilterCountryInput(BaseModel):
    country_name: str = Field(
        description="Country name to filter by (e.g., 'United States', 'France', 'Japan')"
    )


class FindExtremeInput(BaseModel):
    n: int = Field(
        default=5,
        description="Number of years to return"
    )
    extreme_type: str = Field(
        default="warmest",
        description="Type of extreme: 'warmest' or 'coldest'"
    )


class CreateVisualizationInput(BaseModel):
    plot_type: str = Field(
        default="annual_trend",
        description=(
            "Type of plot to create:\n"
            "- 'annual_trend': Shows temperature trend over years (global or for specific country)\n"
            "- 'country_comparison': Automatically finds and displays the TOP 5 COUNTRIES with highest warming trends\n"
            "- 'monthly_pattern': Shows average temperature by month (requires monthly data)"
        )
    )
    country: str = Field(
        default="",
        description="Country name to focus on (only used for 'annual_trend' and 'monthly_pattern'). Leave empty for global. Ignored for 'country_comparison' which always shows top 5."
    )
    save_path: str = Field(
        default="climate_plot.png",
        description="Path to save the plot image"
    )


# Config classes for each tool
class CalculateStatisticsConfig(FunctionBaseConfig, name="calculate_statistics"):
    """Configuration for calculating climate statistics."""
    pass


class ListCountriesConfig(FunctionBaseConfig, name="list_countries"):
    """Configuration for listing available countries."""
    pass


class FilterByCountryConfig(FunctionBaseConfig, name="filter_by_country"):
    """Configuration for filtering by country."""
    pass


class FindExtremeYearsConfig(FunctionBaseConfig, name="find_extreme_years"):
    """Configuration for finding extreme years."""
    pass


class CreateVisualizationConfig(FunctionBaseConfig, name="create_visualization"):
    """Configuration for creating visualizations."""
    pass


# Register tools using clean wrapper pattern
@register_function(config_type=CalculateStatisticsConfig)
async def calculate_statistics_tool(config: CalculateStatisticsConfig, builder: Builder):
    """Register tool for calculating climate statistics."""
    # Load data once at startup
    df = load_climate_data(DATA_PATH)
    
    # 2. Wrapper uses pre-loaded data and ensures string output
    async def _wrapper(country: str = "") -> str:
        # Treat empty string as None for the underlying function
        country_param = None if country == "" else country
        result = calculate_statistics(df, country_param)
        return result  # Already returns JSON string
    
    # 3. Description tells LLM when to use the tool
    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=CalculateStatsInput,
        description=("Calculate temperature statistics globally or for a specific country. "
                     "Returns JSON with: mean_temperature (°C), min_temperature (°C), max_temperature (°C), "
                     "std_deviation (°C), num_records (count), trend_per_decade (°C/decade), "
                     "years_analyzed (e.g. '1950-2025'), and country (if specified).")
    )


@register_function(config_type=ListCountriesConfig)
async def list_countries_tool(config: ListCountriesConfig, builder: Builder):
    """Register tool for listing available countries."""
    df = load_climate_data(DATA_PATH)
    
    async def _wrapper(dummy: str = "") -> str:
        # NAT requires at least one parameter, even if unused
        result = list_countries(df)
        return result  # Already returns JSON string
    
    yield FunctionInfo.from_fn(
        _wrapper,
        description="List all available countries in the climate dataset. Use this when unsure what countries are available."
    )


@register_function(config_type=FilterByCountryConfig)
async def filter_by_country_tool(config: FilterByCountryConfig, builder: Builder):
    """Register tool for filtering by country."""
    df = load_climate_data(DATA_PATH)
    
    async def _wrapper(country_name: str) -> str:
        result = filter_by_country(df, country_name)
        return result  # Already returns JSON string
    
    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=FilterCountryInput,
        description="Get information about climate data for a specific country including number of stations and years covered."
    )


@register_function(config_type=FindExtremeYearsConfig)
async def find_extreme_years_tool(config: FindExtremeYearsConfig, builder: Builder):
    """Register tool for finding extreme years."""
    df = load_climate_data(DATA_PATH)
    
    async def _wrapper(n: int = 5, extreme_type: str = "warmest") -> str:
        result = find_extreme_years(df, n, extreme_type)
        return result  # Already returns JSON string
    
    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=FindExtremeInput,
        description="Find the warmest or coldest years in the global temperature dataset."
    )


@register_function(config_type=CreateVisualizationConfig)
async def create_visualization_tool(config: CreateVisualizationConfig, builder: Builder):
    """Register tool for creating visualizations."""
    df = load_climate_data(DATA_PATH)
    
    async def _wrapper(
        plot_type: str = "annual_trend",
        country: str = "",
        save_path: str = "climate_plot.png"
    ) -> str:
        # Treat empty string as None for the underlying function
        country_param = None if country == "" else country
        result = create_visualization(df, plot_type, country_param, save_path)
        return result  # Already returns string
    
    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=CreateVisualizationInput,
        description=(
            "Create and save climate data visualizations. "
            "For 'country_comparison' plot type, it AUTOMATICALLY finds and visualizes the TOP 5 countries "
            "with highest warming trends - no need to calculate trends separately. "
            "Also creates annual temperature trends and monthly patterns."
        )
    )