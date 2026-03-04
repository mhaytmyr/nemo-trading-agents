import sys
import requests
import json
import pandas as pd


# Imort Climate utilites
from climate_analyzer.utils.climate_tools_simple import (
    load_climate_data,
    calculate_statistics,
    filter_by_country,
    find_extreme_years,
    create_visualization,
    list_countries
)


def test_nat_api_endpoint():

    # Test the API endpoint
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Why central Asia has big desert in the middle explain it with climate"
                }
            ],
            "stream": False
        }
    )

    # Parse and display the response
    if response.status_code == 200:
        result = response.json()
        print(result["choices"][0]["message"]["content"])
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_climate_tools():
    df = load_climate_data("resources/climate_data/temperature_annual.csv")
    print(f"Loaded {len(df)} temperature records")
    print(f"Years covered: {df['year'].min()}-{df['year'].max()}")
    print(f"Countries: {df['country_name'].nunique()}")

    print(calculate_statistics.__doc__)
    print(filter_by_country(df, country_name="Canada"))
    print(find_extreme_years(df))


if __name__ == '__main__':
    #test_nat_api_endpoint()
    test_climate_tools()