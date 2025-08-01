# Plan: Discovering Common Shipping Lanes

## 1. Objective
Identify and visualize the most common shipping routes (lanes) for Capesize vessels using the existing AIS data. The goal is to aggregate thousands of individual vessel tracks into a small number of representative, high-traffic corridors.

## 2. Guiding Principles
- **Leverage Existing Code**: Maximize reuse of `src/data` for loading, `src/utils` for H3 operations, and the configuration system.
- **Configuration-Driven**: The entire process will be controlled by a new set of YAML files (e.g., `lanes_discovery.yaml`).
- **Fit the Architecture**: New logic will be added to `src/`, and a new script in `scripts/` will orchestrate the workflow.
- **Visualization is Key**: The final output must be a clear and interactive map of the discovered lanes.

---

## 3. Proposed Technical Approach

This will be a multi-step process, orchestrated by a new script, `scripts/discover_lanes.py`.

### **Phase 1: Trajectory Extraction**
*   **Goal**: Convert the raw, continuous AIS data into discrete vessel "journeys."
*   **New Logic**: A new module, `src/features/trajectory.py`, will contain functions to:
    1.  Load vessel data using our existing DuckDB/Parquet loaders.
    2.  Group the data by `mmsi` (vessel ID).
    3.  Segment each vessel's history into distinct journeys. A journey ends if there's a time gap greater than a configurable threshold (e.g., 24 hours).
    4.  Represent each journey as an ordered sequence of H3 cells.
*   **Output**: A Parquet file containing `(mmsi, journey_id, h3_sequence)`.

### **Phase 2: Trajectory Clustering**
*   **Goal**: Group similar trajectories together to find common routes. This is the core of the discovery process.
*   **New Logic**: A new module, `src/models/clustering.py`, will implement the clustering algorithm.
    1.  **Similarity Metric**: We need a way to measure the "distance" between two trajectories (H3 sequences). We can start with **Dynamic Time Warping (DTW)**, as it's excellent for comparing sequences of different lengths.
    2.  **Clustering Algorithm**: We will use **DBSCAN**. It's ideal for this task because it doesn't require us to specify the number of clusters (lanes) beforehand. It will automatically find dense regions of similar trajectories.
*   **Output**: The trajectory data from Phase 1, now with an assigned `cluster_id` for each journey.

### **Phase 3: Route Aggregation & Visualization**
*   **Goal**: For each cluster, create a single, representative route and visualize it.
*   **New Logic**: A new module, `src/visualization/lanes.py`, will handle this.
    1.  **Route Centroid**: For each cluster, we'll calculate a "centroid" or average trajectory. This will become our canonical shipping lane.
    2.  **Visualization**: We will use `folium` to plot the discovered lanes on an interactive map. Each lane can be colored by traffic volume (number of journeys in the cluster).
*   **Output**: An interactive `shipping_lanes.html` file in the `visualizations/` directory.

---

## 4. Implementation Plan

### **Step 1: Create New Files & Modules**
-   **Script**: `scripts/discover_lanes.py` (The main orchestrator).
-   **Source Code**:
    -   `src/features/trajectory.py` (For journey extraction).
    -   `src/models/clustering.py` (For DTW and DBSCAN logic).
    -   `src/visualization/lanes.py` (For plotting the final routes).
-   **Configuration**: `config/experiment_configs/lanes_discovery.yaml` (To control all parameters).
-   **Tests**: `tests/test_lanes.py` (To unit test the new logic).

### **Step 2: Add Dependencies**
-   We will need a library for Dynamic Time Warping. The `dtaidistance` package is a fast and well-maintained option. I will add it to `requirements.txt`.

### **Step 3: Develop in Phases**
-   We will implement and test each phase (Trajectory Extraction, Clustering, Visualization) sequentially to ensure each part works correctly before moving to the next.

This approach creates a robust, maintainable, and scalable system for discovering shipping lanes that integrates perfectly with our existing project. What are your thoughts on this plan?
