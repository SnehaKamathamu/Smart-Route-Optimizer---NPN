import os
import numpy as np
import pandas as pd
import requests
import folium
from math import radians, cos, sin, asin, sqrt
import json
from datetime import datetime
import openai
import polyline  # <-- Add this import

OPENROUTE_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"

def haversine_minutes(lat1, lon1, lat2, lon2, kmh=30.0):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    km = 2*6371*asin(sqrt(a))
    if kmh <= 0:
        kmh = 30.0
    minutes = (km/kmh)*60.0
    return minutes

def extract_locations_from_dataset(filepath):
    try:
        if not os.path.exists(filepath):
            print(f"❌ File not found: {filepath}")
            return pd.DataFrame(columns=['latitude', 'longitude', 'location_name'])
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)
        lat_columns = ['latitude', 'lat', 'y', 'coord_y', 'location_lat', 'lat_deg', 'lat_decimal']
        lon_columns = ['longitude', 'lon', 'lng', 'x', 'coord_x', 'location_lon', 'lon_deg', 'lng_decimal', 'long']
        lat_col, lon_col = None, None
        for col in df.columns:
            col_lower = col.lower().strip()
            if not lat_col and any(pat in col_lower for pat in lat_columns):
                lat_col = col
            if not lon_col and any(pat in col_lower for pat in lon_columns):
                lon_col = col
        if lat_col and lon_col:
            df = df.dropna(subset=[lat_col, lon_col])
            df[lat_col] = pd.to_numeric(df[lat_col], errors='coerce')
            df[lon_col] = pd.to_numeric(df[lon_col], errors='coerce')
            df = df.dropna(subset=[lat_col, lon_col])
            df = df[(df[lat_col]>=-90)&(df[lat_col]<=90)]
            df = df[(df[lon_col]>=-180)&(df[lon_col]<=180)]
            if len(df) == 0:
                print("❌ No valid coordinates after validation")
                return pd.DataFrame(columns=['latitude', 'longitude', 'location_name'])
            df['latitude'] = df[lat_col].astype(float)
            df['longitude'] = df[lon_col].astype(float)
            name_columns = ['name', 'location', 'address', 'place', 'store', 'city', 'store_name', 'location_name', 'description']
            name_col = None
            for col in df.columns:
                if col.lower().strip() in [n.lower() for n in name_columns]:
                    name_col = col
                    break
            df['location_name'] = df[name_col].astype(str) if name_col else 'Location ' + (df.index + 1).astype(str)
            result = df[['latitude', 'longitude', 'location_name']].reset_index(drop=True)
            print(f"✅ Extracted {len(result)} locations")
            return result
        else:
            print("❌ No coordinate columns found")
            return pd.DataFrame(columns=['latitude', 'longitude', 'location_name'])
    except Exception as e:
        print(f"❌ Error extracting locations: {e}")
        return pd.DataFrame(columns=['latitude', 'longitude', 'location_name'])

def get_real_time_traffic_data(lat, lon, api_key=None):
    try:
        traffic_factor = np.random.uniform(0.8, 1.5)
        conditions = ["Light Traffic", "Moderate Traffic", "Heavy Traffic", "Congested"]
        condition = np.random.choice(conditions, p=[0.4,0.3,0.2,0.1])
        return {
            "traffic_factor": traffic_factor,
            "condition": condition,
            "speed_factor": 1.0 / traffic_factor,
            "timestamp": datetime.now().isoformat()
        }
    except:
        return {"traffic_factor":1.0, "condition":"Unknown", "speed_factor":1.0}

def get_route_from_ors(start_coords, end_coords, api_key):
    try:
        print("🚦 Using ORS API with key (hidden)")
        headers = {"Authorization": api_key, "Content-Type": "application/json"}
        body = {
            "coordinates": [start_coords, end_coords],
            "format": "geojson",
            "profile": "driving-car"
        }
        response = requests.post(OPENROUTE_DIRECTIONS_URL, json=body, headers=headers, timeout=30)
        print(f"ORS Response status: {response.status_code}")
        print(f"ORS Response preview: {response.text[:500]}")
        if response.status_code == 200:
            data = response.json()
            route = data['routes'][0]
            encoded_geometry = route['geometry']
            coords = polyline.decode(encoded_geometry)  # decode polyline string ➔ list[(lat, lon)]
            return {
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [[lon, lat] for lat, lon in coords]  # convert to [lon, lat]
                },
                'distance': route['summary']['distance'],
                'duration': route['summary']['duration']
            }
        else:
            print(f"ORS API error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"ORS Exception: {e}")
        return None

def get_genai_route_insights(route_data, traffic_data, start_location, end_location):
    try:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key:
            return {
                'insights': "GenAI insights unavailable - OpenAI API key not configured.",
                'recommendations': [
                    "Configure OpenAI API key for AI-powered insights",
                    "Check current traffic conditions before departure",
                    "Consider alternative routes during peak hours"
                ],
                'traffic_analysis': f"Current traffic condition: {traffic_data.get('condition', 'Unknown')}"
            }
        distance_km = route_data.get('distance', 0) / 1000
        duration_min = route_data.get('duration', 0) / 60
        context = f"""
Route Analysis Request:
- Start: {start_location}
- End: {end_location}
- Distance: {distance_km:.1f} km
- Estimated Duration: {duration_min:.0f} minutes
- Current Traffic: {traffic_data.get('condition', 'Unknown')}
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
        except Exception:
            return {
                'insights': ai_response[:300] + "..." if len(ai_response) > 300 else ai_response,
                'recommendations': ["Check traffic conditions", "Consider alternative times", "Plan for potential delays"],
                'traffic_analysis': f"Current conditions: {traffic_data.get('condition', 'Unknown')}"
            }
    except Exception as e:
        return {
            'insights': f"GenAI analysis unavailable: {str(e)}",
            'recommendations': ["Use real-time traffic apps", "Plan for potential delays", "Check road conditions"],
            'traffic_analysis': f"Traffic condition: {traffic_data.get('condition', 'Unknown')}"
        }


def generate_optimized_map(dataset_file, start_index, end_index, output_file="static/map.html", use_genai=True):
    try:
        locations = extract_locations_from_dataset(dataset_file)
        if locations.empty:
            raise ValueError("No locations found in dataset")
        if start_index >= len(locations) or end_index >= len(locations):
            raise ValueError(f"Invalid location indices: start={start_index}, end={end_index}, total={len(locations)}")
        start_location = locations.iloc[start_index]
        end_location = locations.iloc[end_index]
        start_coords = [start_location['longitude'], start_location['latitude']]
        end_coords = [end_location['longitude'], end_location['latitude']]
        print(f"Start coords: {start_coords}, End coords: {end_coords}")
        traffic_data = get_real_time_traffic_data(start_location['latitude'], start_location['longitude'])
        ors_api_key = os.environ.get("ORS_API_KEY")
        route_data = None
        if ors_api_key:
            print("Attempting ORS API route fetch...")
            route_data = get_route_from_ors(start_coords, end_coords, ors_api_key)
        if not route_data or 'geometry' not in route_data or not route_data['geometry'].get('coordinates'):
            print("ORS API failed or no valid route, using fallback straight line.")
            distance = haversine_minutes(start_location['latitude'], start_location['longitude'], end_location['latitude'], end_location['longitude'], kmh=50)*50/60*1000
            duration = haversine_minutes(start_location['latitude'], start_location['longitude'], end_location['latitude'], end_location['longitude'], kmh=50)*60
            duration *= traffic_data.get('traffic_factor',1.0)
            route_data = {
                'distance': distance,
                'duration': duration,
                'geometry': {
                    'type': 'LineString',
                    'coordinates': [start_coords, end_coords]
                }
            }
        genai_insights = None
        if use_genai:
            genai_insights = get_genai_route_insights(route_data, traffic_data, start_location['location_name'], end_location['location_name'])
        center_lat = (start_location['latitude'] + end_location['latitude']) / 2
        center_lon = (start_location['longitude'] + end_location['longitude']) / 2
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        folium.Marker(
            location=[start_location['latitude'], start_location['longitude']],
            popup=f"Start: {start_location['location_name']}",
            icon=folium.Icon(color="green", icon="play")
        ).add_to(m)
        folium.Marker(
            location=[end_location['latitude'], end_location['longitude']],
            popup=f"End: {end_location['location_name']}",
            icon=folium.Icon(color="red", icon="stop")
        ).add_to(m)
        coords = route_data['geometry']['coordinates']
        route_coords = [[coord[1], coord[0]] for coord in coords]  # [lon, lat] -> [lat, lon]
        traffic_condition = traffic_data.get('condition','Unknown')
        color_map = {
            'Light Traffic': 'green',
            'Moderate Traffic': 'orange',
            'Heavy Traffic': 'red',
            'Congested': 'darkred'
        }
        color = color_map.get(traffic_condition, 'blue')
        folium.PolyLine(
            route_coords,
            color=color,
            weight=5,
            opacity=0.8,
            popup=f"Route - {traffic_condition}"
        ).add_to(m)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        m.save(output_file)
        print(f"Map saved to {output_file}")
        return os.path.basename(output_file), route_data, genai_insights
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e
