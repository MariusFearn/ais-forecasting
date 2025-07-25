"""
Ultra-Fast Maritime Chess Board Visualization
Using optimized multi-threading with your 14 cores and 54GB RAM

This approach uses threading (not multiprocessing) to avoid memory conflicts
while maximizing your CPU utilization.
"""

import pickle
import pandas as pd
import h3
import numpy as np
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import time
from collections import defaultdict, Counter
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import warnings
warnings.filterwarnings('ignore')

class FastMaritimeProcessor:
    def __init__(self, max_workers=12):  # Leave 2 cores for system
        self.max_workers = max_workers
        self.global_stats = defaultdict(lambda: {
            'messages': 0,
            'ships': set(),
            'speed_sum': 0,
            'speed_count': 0,
            'draught_sum': 0,
            'draught_count': 0
        })
        self.lock = threading.Lock()
        
    def process_chunk_fast(self, chunk_data):
        """Process a chunk of data with H3 mapping - thread-safe"""
        try:
            # Create H3 mapping for chunk
            chunk_data['h3_cell'] = chunk_data.apply(
                lambda row: h3.geo_to_h3(row['lat'], row['lon'], 6), axis=1
            )
            
            # Aggregate chunk data
            chunk_stats = defaultdict(lambda: {
                'messages': 0,
                'ships': set(),
                'speed_sum': 0,
                'speed_count': 0,
                'draught_sum': 0,
                'draught_count': 0
            })
            
            # Group by H3 cell
            for _, row in chunk_data.iterrows():
                h3_cell = row['h3_cell']
                
                chunk_stats[h3_cell]['messages'] += 1
                chunk_stats[h3_cell]['ships'].add(row['imo'])
                
                if pd.notna(row['speed']):
                    chunk_stats[h3_cell]['speed_sum'] += row['speed']
                    chunk_stats[h3_cell]['speed_count'] += 1
                    
                if pd.notna(row['draught']):
                    chunk_stats[h3_cell]['draught_sum'] += row['draught']
                    chunk_stats[h3_cell]['draught_count'] += 1
            
            return chunk_stats
            
        except Exception as e:
            print(f"   ❌ Chunk processing failed: {e}")
            return None
    
    def merge_stats(self, chunk_stats):
        """Merge chunk statistics into global stats - thread-safe"""
        with self.lock:
            for h3_cell, stats in chunk_stats.items():
                self.global_stats[h3_cell]['messages'] += stats['messages']
                self.global_stats[h3_cell]['ships'].update(stats['ships'])
                self.global_stats[h3_cell]['speed_sum'] += stats['speed_sum']
                self.global_stats[h3_cell]['speed_count'] += stats['speed_count']
                self.global_stats[h3_cell]['draught_sum'] += stats['draught_sum']
                self.global_stats[h3_cell]['draught_count'] += stats['draught_count']
    
    def process_file_threaded(self, file_path, chunk_size=200000):
        """Process a single file using multiple threads"""
        print(f"📊 Processing {file_path.name}...")
        start_time = time.time()
        
        try:
            # Load the entire file (you have 54GB RAM!)
            with open(file_path, 'rb') as f:
                df = pickle.load(f)
            
            load_time = time.time() - start_time
            print(f"   📁 Loaded {len(df):,} records in {load_time:.1f}s")
            
            # Split into chunks for threading
            chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
            print(f"   🔄 Processing {len(chunks)} chunks with {self.max_workers} threads...")
            
            # Process chunks in parallel using threads
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all chunk processing tasks
                future_to_chunk = {
                    executor.submit(self.process_chunk_fast, chunk): i 
                    for i, chunk in enumerate(chunks)
                }
                
                # Collect results as they complete
                completed_chunks = 0
                for future in as_completed(future_to_chunk):
                    chunk_stats = future.result()
                    if chunk_stats:
                        self.merge_stats(chunk_stats)
                        completed_chunks += 1
                        if completed_chunks % 5 == 0:
                            print(f"   ✅ Completed {completed_chunks}/{len(chunks)} chunks")
            
            # Clean up
            del df
            gc.collect()
            
            total_time = time.time() - start_time
            print(f"   🎉 File completed in {total_time:.1f}s ({len(chunks)} chunks)")
            
        except Exception as e:
            print(f"   ❌ File processing failed: {e}")
    
    def process_all_files(self):
        """Process all AIS data files"""
        print("🚢 Ultra-Fast Maritime Chess Board Processing")
        print("=" * 60)
        print(f"💪 Using {self.max_workers} threads with 54GB RAM")
        
        # Find all pickle files
        pickle_files = list(Path("raw_data").glob("*.pkl"))
        print(f"📁 Found {len(pickle_files)} data files")
        
        total_start = time.time()
        
        # Process each file
        for file_path in pickle_files:
            self.process_file_threaded(file_path)
        
        total_time = time.time() - total_start
        print(f"\n🎉 All files processed in {total_time:.1f}s")
        print(f"📊 Total H3 cells with data: {len(self.global_stats):,}")
        
        return self.prepare_results()
    
    def prepare_results(self):
        """Convert global stats to DataFrame for visualization"""
        print("📊 Preparing results for visualization...")
        
        results = []
        for h3_cell, stats in self.global_stats.items():
            # Calculate averages
            avg_speed = (stats['speed_sum'] / stats['speed_count']) if stats['speed_count'] > 0 else 0
            avg_draught = (stats['draught_sum'] / stats['draught_count']) if stats['draught_count'] > 0 else 0
            
            # Get cell center coordinates
            lat, lon = h3.h3_to_geo(h3_cell)
            
            results.append({
                'h3_cell': h3_cell,
                'lat': lat,
                'lon': lon,
                'messages': stats['messages'],
                'ships': len(stats['ships']),
                'avg_speed': avg_speed,
                'avg_draught': avg_draught
            })
        
        df_results = pd.DataFrame(results)
        print(f"✅ Results prepared: {len(df_results):,} active H3 cells")
        
        return df_results

def create_ultra_fast_visualizations(df_results):
    """Create visualizations from processed results"""
    print("\n🎨 Creating ultra-fast visualizations...")
    
    # Create visualizations directory
    Path("visualizations").mkdir(exist_ok=True)
    
    # 1. Global heatmap
    print("   🌍 Creating global maritime heatmap...")
    
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB dark_matter')
    
    # Prepare heatmap data (top 10000 cells for performance)
    top_cells = df_results.nlargest(10000, 'messages')
    heat_data = [[row['lat'], row['lon'], row['messages']] for _, row in top_cells.iterrows()]
    
    HeatMap(
        heat_data,
        radius=12,
        blur=8,
        max_zoom=18,
        gradient={0.2: 'blue', 0.4: 'cyan', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}
    ).add_to(m)
    
    # Add top 20 hotspots
    top_hotspots = df_results.nlargest(20, 'messages')
    for _, hotspot in top_hotspots.iterrows():
        folium.CircleMarker(
            location=[hotspot['lat'], hotspot['lon']],
            radius=8,
            popup=f"""
            <b>Global Maritime Hotspot</b><br>
            Messages: {hotspot['messages']:,}<br>
            Ships: {hotspot['ships']:,}<br>
            Avg Speed: {hotspot['avg_speed']:.1f} knots<br>
            Avg Draught: {hotspot['avg_draught']:.1f}m
            """,
            color='white',
            fillColor='red',
            fillOpacity=0.8
        ).add_to(m)
    
    m.save("visualizations/complete_global_maritime_heatmap.html")
    
    # 2. Interactive 3D globe
    print("   🌐 Creating 3D globe visualization...")
    
    top_100 = df_results.nlargest(100, 'messages')
    
    fig = go.Figure(data=go.Scattergeo(
        lon=top_100['lon'],
        lat=top_100['lat'],
        text=top_100.apply(lambda row: 
            f"Traffic: {row['messages']:,}<br>" +
            f"Ships: {row['ships']:,}<br>" +
            f"Speed: {row['avg_speed']:.1f}kn<br>" +
            f"Draught: {row['avg_draught']:.1f}m", axis=1),
        mode='markers',
        marker=dict(
            size=top_100['messages'] / top_100['messages'].max() * 25 + 5,
            color=top_100['messages'],
            colorscale='Hot',
            colorbar=dict(title="Traffic Intensity"),
            sizemode='diameter',
            opacity=0.8
        )
    ))
    
    fig.update_layout(
        title='Complete Global Maritime Chess Board - 3D View',
        geo=dict(
            projection_type='orthographic',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            showocean=True,
            oceancolor='rgb(0, 100, 200)'
        ),
        width=900,
        height=900
    )
    
    fig.write_html("visualizations/complete_maritime_3d_globe.html")
    
    # 3. Statistics summary
    print("   📊 Creating statistics summary...")
    
    stats_summary = {
        'total_cells': len(df_results),
        'total_messages': df_results['messages'].sum(),
        'total_ships': df_results['ships'].sum(),
        'busiest_cell': df_results.loc[df_results['messages'].idxmax()],
        'avg_messages_per_cell': df_results['messages'].mean(),
        'top_10_cells': df_results.nlargest(10, 'messages')
    }
    
    print(f"\n📈 COMPLETE GLOBAL MARITIME STATISTICS:")
    print(f"   Total active H3 cells: {stats_summary['total_cells']:,}")
    print(f"   Total messages processed: {stats_summary['total_messages']:,}")
    print(f"   Total unique ships: {stats_summary['total_ships']:,}")
    print(f"   Average messages per cell: {stats_summary['avg_messages_per_cell']:.1f}")
    
    print(f"\n🔥 TOP 10 GLOBAL MARITIME HOTSPOTS:")
    for i, (_, row) in enumerate(stats_summary['top_10_cells'].iterrows(), 1):
        print(f"   {i:2d}. {row['messages']:,} messages, {row['ships']:,} ships "
              f"at ({row['lat']:.3f}, {row['lon']:.3f})")
    
    return stats_summary

def main():
    """Main ultra-fast processing function"""
    start_time = time.time()
    
    # Process all data with threading
    processor = FastMaritimeProcessor(max_workers=12)  # Use 12 of your 14 cores
    df_results = processor.process_all_files()
    
    # Create visualizations
    stats = create_ultra_fast_visualizations(df_results)
    
    total_time = time.time() - start_time
    
    print(f"\n" + "=" * 60)
    print(f"🎉 ULTRA-FAST COMPLETE PROCESSING FINISHED!")
    print(f"⏱️  Total time: {total_time:.1f} seconds")
    print(f"🚀 Processing speed: {stats['total_messages']/total_time:,.0f} messages/second")
    print(f"\n📁 Generated visualizations:")
    print(f"   • complete_global_maritime_heatmap.html")
    print(f"   • complete_maritime_3d_globe.html")
    print(f"\n🌊 The complete global maritime chess board is ready!")

if __name__ == "__main__":
    main()
