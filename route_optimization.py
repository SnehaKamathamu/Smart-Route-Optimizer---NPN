# route_optimization_fixed.py - Fixed version with better debugging

import os
import numpy as np
import pandas as pd
import requests
import folium
from math import radians, cos, sin, asin, sqrt
import openai
from datetime import datetime
import json

# ---------------------------
# DEFAULT CONFIG
# ---------------------------
DEFAULT_CITY = "Auto-detect"
NUM_VEHICLES = 1
MAX_ROUTE_DURATION_MIN = 600
ORS_MATRIX_URL = "https://api.openrouteservice.org/v2/matrix/driving-car"
OPENROUTE_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

def haversine_minutes(lat1, lon1, lat2, lon2, kmh=30.0):
    """Calculate travel time between two points using haversine distance"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    km = 2 * 6371 * asin(sqrt(a))
    if kmh <= 0:
        kmh = 30.0
    minutes = (km / kmh) * 60.0
    return minutes

def extract_locations_from_dataset(filepath):
    """
    FIXED: Extract location data from uploaded CSV/Excel file with better debugging.
    """
    print(f"🔍 Starting location extraction from: {filepath}")
    
    try:
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return pd.DataFrame(columns=['latitude', 'longitude', 'location_name'])
        
        # Read file based on extension
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
            print(f"✅ Successfully read CSV file - Shape: {df.shape}")
        else:
            df = pd.read_excel(filepath)
            print(f"✅ Successfully read Excel file - Shape: {df.shape}")
        
        print(f"📋 Available columns: {list(df.columns)}")
        print(f"📄 First row sample: {df.head(1).to_dict('records')}")
        
        # Expanded column name patterns for coordinates
        lat_columns = ['latitude', 'lat', 'y', 'coord_y', 'location_lat', 'lat_deg', 'lat_decimal']
        lon_columns = ['longitude', 'lon', 'lng', 'x', 'coord_x', 'location_lon', 'lon_deg', 'lng_decimal', 'long']
        
        # Find coordinate columns (case insensitive)
        lat_col = None
        lon_col = None
        
        print("🔍 Searching for coordinate columns...")
        for col in df.columns:
            col_lower = col.lower().strip()
            print(f"  Checking column: '{col}' -> '{col_lower}'")
            
            # Check for latitude patterns
            if not lat_col:
                for pattern in lat_columns:
                    if pattern in col_lower:
                        lat_col = col
                        print(f"  ✅ Found latitude column: '{col}' (matched pattern: '{pattern}')")
                        break
            
            # Check for longitude patterns  
            if not lon_col:
                for pattern in lon_columns:
                    if pattern in col_lower:
                        lon_col = col
                        print(f"  ✅ Found longitude column: '{col}' (matched pattern: '{pattern}')")
                        break
        
        if lat_col and lon_col:
            print(f"✅ Using coordinate columns: lat='{lat_col}', lon='{lon_col}'")
            
            # Check data types and sample values
            print(f"📊 Latitude column '{lat_col}' - Type: {df[lat_col].dtype}, Sample: {df[lat_col].head(3).tolist()}")
            print(f"📊 Longitude column '{lon_col}' - Type: {df[lon_col].dtype}, Sample: {df[lon_col].head(3).tolist()}")
            
            # Clean and validate data
            original_count = len(df)
            print(f"📊 Original dataset: {original_count} rows")
            
            # Remove rows with missing coordinates
            df = df.dropna(subset=[lat_col, lon_col])
            print(f"📊 After removing NaN values: {len(df)}/{original_count} rows")
            
            # Convert to numeric if needed
            try:
                df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
                df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
                df = df.dropna(subset=[lat_col, lon_col])
                print(f"📊 After numeric conversion: {len(df)}/{original_count} rows")
            except Exception as e:
                print(f"⚠️ Error converting to numeric: {e}")
            
            # Validate coordinate ranges
            df = df[(df[lat_col] >= -90) & (df[lat_col] <= 90)]
            print(f"📊 After latitude validation (-90 to 90): {len(df)}/{original_count} rows")
            
            df = df[(df[lon_col] >= -180) & (df[lon_col] <= 180)]
            print(f"📊 After longitude validation (-180 to 180): {len(df)}/{original_count} rows")
            
            if len(df) == 0:
                print("❌ No valid coordinates found after validation!")
                return pd.DataFrame(columns=['latitude', 'longitude', 'location_name'])
            
            # Create standardized columns
            df['latitude'] = df[lat_col].astype(float)
            df['longitude'] = df[lon_col].astype(float)
            
            # Find name/description column
            name_columns = ['name', 'location', 'address', 'place', 'store', 'city', 'store_name', 'location_name', 'description']
            name_col = None
            
            print("🔍 Searching for name column...")
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [nc.lower() for nc in name_columns]:
                    name_col = col
                    print(f"✅ Found name column: '{col}'")
                    break
            
            if name_col:
                df['location_name'] = df[name_col].astype(str)
            else:
                print("⚠️ No name column found, generating generic names")
                df['location_name'] = 'Location ' + (df.index + 1).astype(str)
            
            # Create final result
            result = df[['latitude', 'longitude', 'location_name']].copy()
            result = result.reset_index(drop=True)
            
            print(f"✅ Successfully extracted {len(result)} locations")
            print(f"📋 Sample locations:")
            for i, row in result.head(3).iterrows():
                print(f"  {i}: {row['location_name']} ({row['latitude']:.4f}, {row['longitude']:.4f})")
            
            return result
            
        else:
            print("❌ No coordinate columns found!")
            print(f"   Available columns: {list(df.columns)}")
            print(f"   Looked for latitude patterns: {lat_columns}")
            print(f"   Looked for longitude patterns: {lon_columns}")
            
            # Try to show similar column names
            print("🔍 Possible coordinate columns:")
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['lat', 'lon', 'lng', 'coord', 'x', 'y']):
                    print(f"   - '{col}' (might contain coordinates)")
            
            return pd.DataFrame(columns=['latitude', 'longitude', 'location_name'])
    
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['latitude', 'longitude', 'location_name'])

def get_real_time_traffic_data(lat, lon, api_key=None):
    """Fetch simulated real-time traffic data"""
    try:
        traffic_factor = np.random.uniform(0.8, 1.5)
        conditions = ["Light Traffic", "Moderate Traffic", "Heavy Traffic", "Congested"]
        condition = np.random.choice(conditions, p=[0.4, 0.3, 0.2, 0.1])
        
        return {
            'traffic_factor': traffic_factor,
            'condition': condition,
            'speed_factor': 1.0 / traffic_factor,
            'timestamp': datetime.now().isoformat()
        }
    except:
        return {'traffic_factor': 1.0, 'condition': 'Unknown', 'speed_factor': 1.0}

def get_route_from_ors(start_coords, end_coords, api_key):
    """Get optimized route from OpenRouteService"""
    try:
        headers = {"Authorization": api_key, "Content-Type": "application/json"}
        body = {
            "coordinates": [start_coords, end_coords],
            "format": "geojson",
            "profile": "driving-car",
            "options": {"avoid_features": ["ferries"]},
            "preference": "fastest"
        }
        
        response = requests.post(OPENROUTE_DIRECTIONS_URL, json=body, headers=headers, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            route = data['features'][0]
            return {
                'geometry': route['geometry'],
                'distance': route['properties']['summary']['distance'],
                'duration': route['properties']['summary']['duration']
            }
        else:
            print(f"⚠️ ORS API error: {response.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ ORS API error: {e}")
        return None

def get_genai_route_insights(route_data, traffic_data, start_location, end_location):
    """Generate AI-powered insights about the route using OpenAI"""
    try:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return {
                'insights': "GenAI insights unavailable - OpenAI API key not configured. Add your OpenAI API key to the .env file to enable AI-powered route analysis.",
                'recommendations': [
                    "Configure OpenAI API key for AI-powered insights",
                    "Check current traffic conditions before departure",
                    "Consider alternative routes during peak hours"
                ],
                'traffic_analysis': f"Current traffic condition: {traffic_data.get('condition', 'Unknown')}"
            }
        
        # Prepare context for AI
        distance_km = route_data.get('distance', 0) / 1000
        duration_min = route_data.get('duration', 0) / 60
        
        context = f"""
        Route Analysis Request:
        - Start: {start_location}
        - End: {end_location}
        - Distance: {distance_km:.1f} km
        - Estimated Duration: {duration_min:.0f} minutes
        - Current Traffic: {traffic_data.get('condition', 'Unknown')}
        - Traffic Factor: {traffic_data.get('traffic_factor', 1.0):.2f}
        """
        
        client = openai.OpenAI(api_key=openai_api_key)
        
        prompt = f"""
        As a logistics and transportation expert, analyze this route and provide insights:
        
        {context}
        
        Please provide:
        1. Route efficiency analysis
        2. Traffic impact assessment
        3. Optimization recommendations
        4. Alternative suggestions
        5. Estimated fuel consumption and costs
        
        Format your response as JSON with keys: insights, recommendations, traffic_analysis, fuel_estimate, cost_estimate.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert logistics analyst providing route optimization insights."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        try:
            insights = json.loads(ai_response)
            return insights
        except:
            return {
                'insights': ai_response[:300] + "..." if len(ai_response) > 300 else ai_response,
                'recommendations': ["Check traffic conditions", "Consider alternative times", "Plan for potential delays"],
                'traffic_analysis': f"Current conditions: {traffic_data.get('condition', 'Unknown')}"
            }
        
    except Exception as e:
        print(f"⚠️ GenAI error: {e}")
        return {
            'insights': f"GenAI analysis unavailable: {str(e)}",
            'recommendations': ["Use real-time traffic apps", "Plan for potential delays", "Check road conditions"],
            'traffic_analysis': f"Traffic condition: {traffic_data.get('condition', 'Unknown')}"
        }

def generate_optimized_map(dataset_file, start_index, end_index, output_file="static/map.html", use_genai=True):
    """Generate optimized route map between two selected points from uploaded dataset"""
    print(f"🚀 Starting route generation: {dataset_file}, start={start_index}, end={end_index}")
    
    try:
        # Load locations
        locations = extract_locations_from_dataset(dataset_file)
        print(f"📊 Loaded {len(locations)} locations")
        
        if locations.empty:
            raise ValueError("No locations found in dataset")
        
        if start_index >= len(locations) or end_index >= len(locations):
            raise ValueError(f"Invalid location indices: start={start_index}, end={end_index}, total={len(locations)}")
        
        start_location = locations.iloc[start_index]
        end_location = locations.iloc[end_index]
        
        print(f"🚀 Start: {start_location['location_name']} ({start_location['latitude']}, {start_location['longitude']})")
        print(f"🏁 End: {end_location['location_name']} ({end_location['latitude']}, {end_location['longitude']})")
        
        start_coords = [start_location['longitude'], start_location['latitude']]
        end_coords = [end_location['longitude'], end_location['latitude']]
        
        # Get real-time traffic data
        traffic_data = get_real_time_traffic_data(start_location['latitude'], start_location['longitude'])
        print(f"🚦 Traffic condition: {traffic_data['condition']}")
        
        # Try to get route from ORS API
        ors_api_key = os.environ.get("ORS_API_KEY")
        route_data = None
        
        if ors_api_key:
            print("🗺️ Trying OpenRouteService API...")
            route_data = get_route_from_ors(start_coords, end_coords, ors_api_key)
        
        # Fallback: calculate direct route
        if not route_data:
            print("🗺️ Using fallback route calculation...")
            distance = haversine_minutes(
                start_location['latitude'], start_location['longitude'],
                end_location['latitude'], end_location['longitude'], kmh=50
            ) * 50 / 60 * 1000  # Convert to meters
            
            duration = haversine_minutes(
                start_location['latitude'], start_location['longitude'],
                end_location['latitude'], end_location['longitude'], kmh=50
            ) * 60  # Convert to seconds
            
            # Apply traffic factor
            duration *= traffic_data.get('traffic_factor', 1.0)
            
            route_data = {
                'distance': distance,
                'duration': duration,
                'geometry': {
                    'coordinates': [start_coords, end_coords]
                }
            }
        
        print(f"📏 Route distance: {route_data['distance']/1000:.1f} km")
        print(f"⏱️ Route duration: {route_data['duration']/60:.0f} minutes")
        
        # Generate GenAI insights
        genai_insights = None
        if use_genai:
            print("🤖 Generating AI insights...")
            genai_insights = get_genai_route_insights(
                route_data, traffic_data, 
                start_location['location_name'], 
                end_location['location_name']
            )
        
        # Create map
        center_lat = (start_location['latitude'] + end_location['latitude']) / 2
        center_lon = (start_location['longitude'] + end_location['longitude']) / 2
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Add start marker
        folium.Marker(
            location=[start_location['latitude'], start_location['longitude']],
            popup=f"Start: {start_location['location_name']}",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        
        # Add end marker  
        folium.Marker(
            location=[end_location['latitude'], end_location['longitude']],
            popup=f"End: {end_location['location_name']}",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)
        
        # Add route line
        if 'coordinates' in route_data['geometry']:
            coords = route_data['geometry']['coordinates']
            if len(coords) == 2:  # Direct line
                route_coords = [[coords[0][1], coords[0][0]], [coords[1][1], coords[1][0]]]
            else:  # Full route from API
                route_coords = [[coord[1], coord[0]] for coord in coords]
            
            # Color based on traffic
            traffic_condition = traffic_data.get('condition', 'Unknown')
            color = {'Light Traffic': 'green', 'Moderate Traffic': 'orange', 
                    'Heavy Traffic': 'red', 'Congested': 'darkred'}.get(traffic_condition, 'blue')
            
            folium.PolyLine(
                route_coords, 
                color=color, 
                weight=5, 
                opacity=0.8,
                popup=f"Route - {traffic_condition}"
            ).add_to(m)
        
        # Save map
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        m.save(output_file)
        print(f"✅ Map saved to: {output_file}")
        
        return os.path.basename(output_file), route_data, genai_insights
        
    except Exception as e:
        print(f"❌ Error generating map: {e}")
        import traceback
        traceback.print_exc()
        raise e