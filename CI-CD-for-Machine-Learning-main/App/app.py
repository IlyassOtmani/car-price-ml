import gradio as gr
import skops.io as sio
import pandas as pd

# Load the trained pipeline with proper security handling
model_path = "../Model/model_pipeline.skops"
unknown_types = sio.get_untrusted_types(file=model_path)
pipe = sio.load(model_path, trusted=unknown_types)


def predict(symboling, wheelbase, carlength, carwidth, carheight, curbweight,
            enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm,
            citympg, highwaympg, fueltype, aspiration, doornumber, carbody,
            drivewheel, enginelocation, enginetype, cylindernumber, fuelsystem, carbrand):
    """
    Make car price predictions based on input features.
    """
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'symboling': [symboling],
        'wheelbase': [wheelbase],
        'carlength': [carlength],
        'carwidth': [carwidth],
        'carheight': [carheight],
        'curbweight': [curbweight],
        'enginesize': [enginesize],
        'boreratio': [boreratio],
        'stroke': [stroke],
        'compressionratio': [compressionratio],
        'horsepower': [horsepower],
        'peakrpm': [peakrpm],
        'citympg': [citympg],
        'highwaympg': [highwaympg],
        'fueltype': [fueltype],
        'aspiration': [aspiration],
        'doornumber': [doornumber],
        'carbody': [carbody],
        'drivewheel': [drivewheel],
        'enginelocation': [enginelocation],
        'enginetype': [enginetype],
        'cylindernumber': [cylindernumber],
        'fuelsystem': [fuelsystem],
        'CarBrand': [carbrand]
    })

    # Make prediction
    prediction = pipe.predict(input_data)[0]

    return f"${prediction:,.2f}"


# Configure inputs
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚗 Car Price Predictor")
    gr.Markdown("""
    Enter car specifications to predict its price. This model uses Random Forest Regression 
    trained on 200+ car samples with various specifications.
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📊 Risk & Insurance")
            symboling = gr.Slider(-3, 3, step=1, value=0, label="Insurance Risk Rating (-3=Safe, 3=Risky)")
            
            gr.Markdown("### 🔧 Engine Specifications")
            enginesize = gr.Slider(60, 330, step=5, value=130, label="Engine Size (cu. inches)")
            horsepower = gr.Slider(48, 288, step=5, value=110, label="Horsepower")
            peakrpm = gr.Slider(4000, 6600, step=100, value=5000, label="Peak RPM")
            boreratio = gr.Slider(2.5, 4.0, step=0.1, value=3.3, label="Bore Ratio")
            stroke = gr.Slider(2.0, 4.5, step=0.1, value=3.0, label="Stroke")
            compressionratio = gr.Slider(7, 23, step=0.5, value=10, label="Compression Ratio")
            
            gr.Markdown("### ⛽ Fuel & Performance")
            citympg = gr.Slider(13, 49, step=1, value=25, label="City MPG")
            highwaympg = gr.Slider(16, 54, step=1, value=30, label="Highway MPG")
            fueltype = gr.Dropdown(["gas", "diesel"], value="gas", label="Fuel Type")
            aspiration = gr.Dropdown(["std", "turbo"], value="std", label="Aspiration")
            fuelsystem = gr.Dropdown(
                ["mpfi", "2bbl", "idi", "1bbl", "spdi", "4bbl", "mfi", "spfi"],
                value="mpfi",
                label="Fuel System"
            )
        
        with gr.Column():
            gr.Markdown("### 📏 Dimensions & Weight")
            wheelbase = gr.Slider(86, 120, step=0.5, value=100, label="Wheelbase (inches)")
            carlength = gr.Slider(141, 208, step=1, value=175, label="Car Length (inches)")
            carwidth = gr.Slider(60, 73, step=0.5, value=66, label="Car Width (inches)")
            carheight = gr.Slider(47, 60, step=0.5, value=54, label="Car Height (inches)")
            curbweight = gr.Slider(1488, 4066, step=50, value=2500, label="Curb Weight (lbs)")
            
            gr.Markdown("### 🚘 Body & Configuration")
            carbrand = gr.Dropdown(
                ["toyota", "nissan", "mazda", "mitsubishi", "honda", "volkswagen", 
                 "subaru", "volvo", "bmw", "mercedes-benz", "audi", "porsche", 
                 "jaguar", "alfa-romeo", "peugeot", "renault", "saab", "dodge", 
                 "plymouth", "chevrolet", "buick", "mercury", "isuzu"],
                value="toyota",
                label="Car Brand"
            )
            carbody = gr.Dropdown(
                ["sedan", "hatchback", "wagon", "hardtop", "convertible"],
                value="sedan",
                label="Body Type"
            )
            doornumber = gr.Dropdown(["two", "four"], value="four", label="Number of Doors")
            drivewheel = gr.Dropdown(["rwd", "fwd", "4wd"], value="fwd", label="Drive Wheel")
            enginelocation = gr.Dropdown(["front", "rear"], value="front", label="Engine Location")
            enginetype = gr.Dropdown(
                ["dohc", "ohc", "ohcv", "l", "rotor", "ohcf", "dohcv"],
                value="ohc",
                label="Engine Type"
            )
            cylindernumber = gr.Dropdown(
                ["four", "six", "five", "eight", "two", "twelve", "three"],
                value="four",
                label="Number of Cylinders"
            )
    
    with gr.Row():
        predict_btn = gr.Button("🔮 Predict Price", variant="primary", size="lg")
    
    with gr.Row():
        output = gr.Textbox(label="Predicted Car Price", scale=1)
    
    predict_btn.click(
        fn=predict,
        inputs=[
            symboling, wheelbase, carlength, carwidth, carheight, curbweight,
            enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm,
            citympg, highwaympg, fueltype, aspiration, doornumber, carbody,
            drivewheel, enginelocation, enginetype, cylindernumber, fuelsystem, carbrand
        ],
        outputs=output
    )
    
    # Example inputs
    gr.Markdown("### 📝 Example Configurations")
    gr.Examples(
        examples=[
            # Economy car - Toyota Corolla-like
            [0, 98.8, 168.8, 64.1, 52.4, 2385, 130, 3.47, 2.68, 9.0, 102, 5500, 24, 30, "gas", "std", "four", "sedan", "fwd", "front", "ohc", "four", "mpfi", "toyota"],
            # Luxury sedan - BMW-like
            [1, 110.0, 193.8, 70.9, 56.3, 3740, 209, 3.62, 3.39, 8.0, 182, 5400, 16, 22, "gas", "std", "four", "sedan", "rwd", "front", "ohc", "six", "mpfi", "bmw"],
            # Sports car - Porsche-like
            [2, 89.5, 168.9, 68.3, 50.2, 2778, 194, 3.74, 3.15, 9.5, 207, 5900, 17, 25, "gas", "turbo", "two", "hardtop", "rwd", "rear", "ohcf", "six", "mpfi", "porsche"]
        ],
        inputs=[
            symboling, wheelbase, carlength, carwidth, carheight, curbweight,
            enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm,
            citympg, highwaympg, fueltype, aspiration, doornumber, carbody,
            drivewheel, enginelocation, enginetype, cylindernumber, fuelsystem, carbrand
        ],
        label="Try these sample cars"
    )
    
    gr.Markdown("""
    ### 🔍 Feature Guide
    
    *Insurance Risk Rating (Symboling)*
    - -3 to -1: Low risk (safer cars)
    - 0: Average risk
    - 1 to 3: High risk (riskier/sportier cars)
    
    *Body Types*
    - *Sedan*: Traditional 4-door car
    - *Hatchback*: Compact car with rear door
    - *Wagon*: Station wagon
    - *Hardtop*: Coupe-style
    - *Convertible*: Retractable roof
    
    *Drive Wheel*
    - *FWD*: Front-wheel drive (fuel efficient)
    - *RWD*: Rear-wheel drive (performance)
    - *4WD*: Four-wheel drive (all-terrain)
    
    *Engine Types*
    - *OHC*: Overhead cam (most common)
    - *DOHC*: Dual overhead cam (performance)
    - *L*: Inline engine
    - *OHCV*: OHC V-type
    
    ### 📈 About the Model
    - *Algorithm*: Random Forest Regressor
    - *Training Data*: 200+ car samples
    - *Features*: 24 car specifications
    - *Deployment*: Automated CI/CD with GitHub Actions
    """)

# Launch app
demo.launch()