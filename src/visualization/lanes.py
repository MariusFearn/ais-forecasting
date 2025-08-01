"""
Functions for visualizing discovered shipping lanes and terminals.
"""
import folium
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any

def load_terminals_and_routes(terminals_path: str, routes_path: str) -> tuple:
    """
    Loads terminal and route data from GeoPackage files.

    Args:
        terminals_path (str): Path to terminals GeoPackage file.
        routes_path (str): Path to routes GeoPackage file.

    Returns:
        tuple: (terminals_gdf, routes_gdf) GeoDataFrames.
    """
    try:
        terminals_gdf = gpd.read_file(terminals_path)
        logging.info(f"Loaded {len(terminals_gdf)} terminals from {terminals_path}")
    except Exception as e:
        logging.error(f"Error loading terminals: {e}")
        terminals_gdf = gpd.GeoDataFrame()

    try:
        routes_gdf = gpd.read_file(routes_path)
        logging.info(f"Loaded {len(routes_gdf)} routes from {routes_path}")
    except Exception as e:
        logging.error(f"Error loading routes: {e}")
        routes_gdf = gpd.GeoDataFrame()

    return terminals_gdf, routes_gdf

def create_base_map(config: dict) -> folium.Map:
    """
    Creates a base folium map with appropriate center and zoom.

    Args:
        config (dict): Configuration dictionary with map settings.

    Returns:
        folium.Map: Base map object.
    """
    # Default to global view if no specific location provided
    start_location = config.get('map_start_location', [0, 0])
    zoom_start = config.get('map_zoom_start', 2)

    # Create base map
    base_map = folium.Map(
        location=start_location,
        zoom_start=zoom_start,
        tiles='OpenStreetMap'
    )

    # Add additional tile layers
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(base_map)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(base_map)
    
    return base_map

def add_terminals_to_map(folium_map: folium.Map, terminals_gdf: gpd.GeoDataFrame) -> None:
    """
    Adds terminal markers to the map.

    Args:
        folium_map (folium.Map): The folium map object.
        terminals_gdf (gpd.GeoDataFrame): GeoDataFrame of terminal points.
    """
    if terminals_gdf.empty:
        logging.warning("No terminals to add to map")
        return

    # Create a feature group for terminals
    terminal_group = folium.FeatureGroup(name="Terminals", show=True)

    for _, terminal in terminals_gdf.iterrows():
        # Get coordinates
        lat = terminal.geometry.y
        lon = terminal.geometry.x

        # Create popup text with terminal information
        popup_text = f"""
        <b>Terminal {terminal['terminal_id']}</b><br>
        Total Visits: {terminal['total_visits']}<br>
        Unique Vessels: {terminal['unique_vessels']}<br>
        Start Visits: {terminal['start_visits']}<br>
        End Visits: {terminal['end_visits']}<br>
        First Visit: {terminal['first_visit']}<br>
        Last Visit: {terminal['last_visit']}
        """

        # Size marker based on activity level
        marker_size = min(20, max(5, terminal['total_visits'] / 10))

        # Create marker
        folium.CircleMarker(
            location=[lat, lon],
            radius=marker_size,
            popup=folium.Popup(popup_text, max_width=300),
            color='red',
            fillColor='darkred',
            fillOpacity=0.7,
            weight=2
        ).add_to(terminal_group)

    terminal_group.add_to(folium_map)
    logging.info(f"Added {len(terminals_gdf)} terminals to map")

def add_routes_to_map(folium_map: folium.Map, routes_gdf: gpd.GeoDataFrame) -> None:
    """
    Adds route lines to the map, styled by traffic volume.

    Args:
        folium_map (folium.Map): The folium map object.
        routes_gdf (gpd.GeoDataFrame): GeoDataFrame of route linestrings.
    """
    if routes_gdf.empty:
        logging.warning("No routes to add to map")
        return

    # Create a feature group for routes
    route_group = folium.FeatureGroup(name="Shipping Lanes", show=True)

    # Calculate quantiles for color coding
    if 'total_journeys' in routes_gdf.columns:
        traffic_quantiles = routes_gdf['total_journeys'].quantile([0.25, 0.5, 0.75])
    else:
        traffic_quantiles = pd.Series([1, 5, 10])

    for _, route in routes_gdf.iterrows():
        # Extract coordinates from LineString geometry
        coords = [[point[1], point[0]] for point in route.geometry.coords]  # Flip to lat,lon

        # Determine line style based on traffic volume
        traffic = route.get('total_journeys', 1)
        if traffic >= traffic_quantiles[0.75]:
            color = 'red'
            weight = 4
            opacity = 0.8
        elif traffic >= traffic_quantiles[0.5]:
            color = 'orange'
            weight = 3
            opacity = 0.7
        elif traffic >= traffic_quantiles[0.25]:
            color = 'yellow'
            weight = 2
            opacity = 0.6
        else:
            color = 'blue'
            weight = 1
            opacity = 0.5

        # Create popup text with route information
        popup_text = f"""
        <b>Route {route['route_id']}</b><br>
        Total Journeys: {route.get('total_journeys', 'N/A')}<br>
        Unique Vessels: {route.get('unique_vessels', 'N/A')}<br>
        Avg Duration: {route.get('avg_duration_hours', 'N/A'):.1f} hours<br>
        Start Terminal: {route.get('start_terminal_id', 'N/A')}<br>
        End Terminal: {route.get('end_terminal_id', 'N/A')}<br>
        First Journey: {route.get('first_journey', 'N/A')}<br>
        Last Journey: {route.get('last_journey', 'N/A')}
        """

        # Create route line
        folium.PolyLine(
            locations=coords,
            popup=folium.Popup(popup_text, max_width=350),
            color=color,
            weight=weight,
            opacity=opacity
        ).add_to(route_group)

    route_group.add_to(folium_map)
    logging.info(f"Added {len(routes_gdf)} routes to map")

def add_legend_to_map(folium_map: folium.Map) -> None:
    """
    Adds a legend explaining the map symbols.

    Args:
        folium_map (folium.Map): The folium map object.
    """
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px
                ">
    <p><b>Shipping Lanes Legend</b></p>
    <p><i class="fa fa-circle" style="color:red"></i> Terminals (Ports/Anchorages)</p>
    <p><i class="fa fa-minus" style="color:red"></i> High Traffic Routes</p>
    <p><i class="fa fa-minus" style="color:orange"></i> Medium Traffic Routes</p>
    <p><i class="fa fa-minus" style="color:blue"></i> Low Traffic Routes</p>
    </div>
    '''
    folium_map.get_root().html.add_child(folium.Element(legend_html))

def plot_shipping_lanes_map(terminals_gdf: gpd.GeoDataFrame, routes_gdf: gpd.GeoDataFrame, config: dict) -> folium.Map:
    """
    Creates an interactive map of shipping terminals and routes.

    Args:
        terminals_gdf (gpd.GeoDataFrame): GeoDataFrame of terminal points.
        routes_gdf (gpd.GeoDataFrame): GeoDataFrame of route linestrings.
        config (dict): Configuration dictionary with map settings.

    Returns:
        folium.Map: An interactive folium map object.
    """
    # Create base map
    folium_map = create_base_map(config)

    # Add terminals
    add_terminals_to_map(folium_map, terminals_gdf)

    # Add routes
    add_routes_to_map(folium_map, routes_gdf)

    # Add legend
    add_legend_to_map(folium_map)

    # Add layer control
    folium.LayerControl().add_to(folium_map)

    logging.info("Created interactive shipping lanes map")
    return folium_map

def save_map_to_html(folium_map: folium.Map, output_path: str) -> None:
    """
    Saves the folium map to an HTML file.

    Args:
        folium_map (folium.Map): The folium map object.
        output_path (str): Path to save the HTML file.
    """
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save map
    folium_map.save(str(output_path))
    logging.info(f"Saved interactive map to {output_path}")

def create_and_save_shipping_lanes_map(config: dict) -> None:
    """
    Complete workflow to create and save the shipping lanes visualization.

    Args:
        config (dict): Configuration dictionary with all paths and settings.
    """
    # Load data
    terminals_path = config['terminals']['output_path']
    routes_path = config['graph']['output_path']
    
    terminals_gdf, routes_gdf = load_terminals_and_routes(terminals_path, routes_path)

    # Create map
    folium_map = plot_shipping_lanes_map(terminals_gdf, routes_gdf, config['visualization'])

    # Save map
    output_path = config['visualization']['map_output_path']
    save_map_to_html(folium_map, output_path)

    logging.info("Shipping lanes visualization workflow completed")
