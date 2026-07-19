//! Example weather tools that calls the [OpenMeteo API](https://open-meteo.com/).

use llama_cpp_2::model::LlamaChatTool;
use serde::Deserialize;

#[derive(Deserialize)]
struct WeatherToolArguments {
    latitude: f64,
    longitude: f64,
}

#[derive(Deserialize)]
struct GeocodeToolArguments {
    city: String,
}

pub fn get_weather_tool_definition() -> Result<LlamaChatTool, std::ffi::NulError> {
    let parameters = serde_json::json!({
        "type": "object",
        "properties": {
            "latitude": {
                "type": "number",
                "description": "Latitude of the location"
            },
            "longitude": {
                "type": "number",
                "description": "Longitude of the location"
            }
        },
        "required": ["latitude", "longitude"]
    });

    LlamaChatTool::new(
        "get_current_weather".into(),
        "Get the current weather at a given latitude and longitude".into(),
        parameters.to_string(),
    )
}

pub fn get_geocode_tool_definition() -> Result<LlamaChatTool, std::ffi::NulError> {
    let parameters = serde_json::json!({
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "The name of the city to get the coordinates for (e.g. 'Santa Barbara')"
            }
        },
        "required": ["city"]
    });

    LlamaChatTool::new(
        "geocode_city".into(),
        "Get the latitude and longitude for a given city".into(),
        parameters.to_string(),
    )
}

pub fn execute_weather(arguments_json: &str) -> String {
    println!("[DEBUG] execute_weather called with arguments: {}", arguments_json);
    let args: WeatherToolArguments = match serde_json::from_str(arguments_json) {
        Ok(a) => a,
        Err(e) => {
            println!("[DEBUG] execute_weather parsing failed: {}", e);
            return format!("Failed to parse arguments: {}", e);
        }
    };

    let weather_url = format!(
        "https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}&current_weather=true",
        args.latitude, args.longitude
    );
    println!("[DEBUG] execute_weather requesting URL: {}", weather_url);

    match reqwest::blocking::get(&weather_url) {
        Ok(response) => match response.text() {
            Ok(text) => {
                println!("[DEBUG] execute_weather success");
                text
            }
            Err(e) => {
                println!("[DEBUG] execute_weather failed to read text: {}", e);
                format!("Failed to read weather response: {}", e)
            }
        },
        Err(e) => {
            println!("[DEBUG] execute_weather HTTP request failed: {}", e);
            format!("Weather request failed: {}", e)
        }
    }
}

pub fn execute_geocode(arguments_json: &str) -> String {
    println!("[DEBUG] execute_geocode called with arguments: {}", arguments_json);
    let args: GeocodeToolArguments = match serde_json::from_str(arguments_json) {
        Ok(a) => a,
        Err(e) => {
            println!("[DEBUG] execute_geocode parsing failed: {}", e);
            return format!("Failed to parse arguments: {}", e);
        }
    };

    let city_query = args.city.replace(" ", "+");
    let geo_url = format!("https://geocoding-api.open-meteo.com/v1/search?name={}&count=1&language=en&format=json", city_query);
    println!("[DEBUG] execute_geocode requesting URL: {}", geo_url);

    let geo_resp = match reqwest::blocking::get(&geo_url) {
        Ok(resp) => resp,
        Err(e) => {
            println!("[DEBUG] execute_geocode HTTP request failed: {}", e);
            return format!("Geocoding request failed: {}", e);
        }
    };

    let text = match geo_resp.text() {
        Ok(t) => t,
        Err(e) => {
            println!("[DEBUG] execute_geocode failed to read text: {}", e);
            return format!("Failed to read geocoding response: {}", e);
        }
    };

    let geo_json: serde_json::Value = match serde_json::from_str(&text) {
        Ok(j) => j,
        Err(e) => {
            println!("[DEBUG] execute_geocode JSON parse failed: {}", e);
            return format!("Failed to parse geocoding JSON: {}", e);
        }
    };

    let results = geo_json.get("results").and_then(|r| r.as_array());
    if results.is_none() || results.unwrap().is_empty() {
        println!("[DEBUG] execute_geocode found no results for city: {}", args.city);
        return format!("City '{}' not found.", args.city);
    }

    let location = &results.unwrap()[0];
    let lat = location["latitude"].as_f64().unwrap_or(0.0);
    let lon = location["longitude"].as_f64().unwrap_or(0.0);

    println!("[DEBUG] execute_geocode success: lat={}, lon={}", lat, lon);

    serde_json::json!({
        "latitude": lat,
        "longitude": lon
    }).to_string()
}
