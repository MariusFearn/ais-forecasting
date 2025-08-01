# Shipping Lane Discovery: To-Do List

This checklist tracks the implementation of the hybrid plan outlined in `make_route_plan.md`.

## Phase 1: Trajectory & Terminal Extraction

-   [x] **Create New Files & Modules**
    -   [x] `scripts/discover_lanes.py` (main script)
    -   [x] `src/features/trajectory.py`
    -   [x] `src/models/clustering.py`
    -   [x] `src/visualization/lanes.py`
    -   [x] `tests/test_lanes.py`
-   [x] **Create Configuration File**
    -   [x] `config/experiment_configs/lanes_discovery.yaml`
-   [x] **Implement Trajectory Segmentation** (`src/features/trajectory.py`)
    -   [x] Function to load data for a single vessel.
    -   [x] Function to segment vessel data into journeys based on time gaps.
    -   [x] Function to convert journeys into H3 sequences.
    -   [x] Main function to process all vessels and save journeys to a Parquet file.
-   [x] **Implement Terminal Clustering** (`src/models/clustering.py`)
    -   [x] Function to extract all start/end points from the journeys file.
    -   [x] Function to cluster these points using DBSCAN to find terminals.
    -   [x] Function to save terminal clusters to a GeoPackage file.

## Phase 2: Route Clustering

-   [x] **Implement Trajectory Distance Matrix** (`src/models/clustering.py`)
    -   [x] Function to load the H3 journey sequences.
    -   [x] Function to compute the pairwise DTW distance between all journey sequences.
-   [x] **Implement Route Clustering** (`src/models/clustering.py`)
    -   [x] Function to run DBSCAN on the pre-computed distance matrix.
    -   [x] Function to assign route cluster IDs back to the journey data.
    -   [x] Function to save the clustered journeys to a new Parquet file.

## Phase 3: Route Graph Construction

-   [x] **Implement Route Centroid Calculation** (`src/features/trajectory.py`)
    -   [x] Function to calculate the "centroid" (representative path) for each route cluster.
-   [x] **Implement Graph Assembly** (`src/features/trajectory.py`)
    -   [x] Function to link each route centroid to its most likely start and end terminals.
    -   [x] Function to gather metadata for each route (e.g., traffic volume, average journey time).
-   [x] **Implement Graph Saving**
    -   [x] Function to save the final route graph (edges) to a GeoPackage file.

## Phase 4: Validation & Visualization

-   [x] **Implement Validation Metrics** (`src/utils/metrics.py`)
    -   [x] Function to calculate clustering metrics (e.g., silhouette score, if applicable).
    -   [x] Function to log the number of clusters, outliers, and cluster sizes.
-   [x] **Implement Visualization** (`src/visualization/lanes.py`)
    -   [x] Function to load the terminal (nodes) and route (edges) GeoPackage files.
    -   [x] Function to create an interactive `folium` map.
    -   [x] Function to plot terminals as markers and routes as lines.
    -   [x] Function to style routes based on traffic volume.
    -   [x] Function to save the map to `visualizations/shipping_lanes.html`.

## Final Step: Integration

-   [x] **Update Main Script** (`scripts/discover_lanes.py`)
    -   [x] Integrate all phases into a single, configuration-driven pipeline.
    -   [x] Add logging and clear status updates for each step.
    -   [x] Add command-line argument parsing for the config file.
