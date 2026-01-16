import gradio as gr
import numpy as np
import pickle
import pandas as pd


# load the model
with open(
r"D:\Laptop Price Prediction & Feature Importance Analysis\random_forest_model.plk","rb") as file:
    model = pickle.load(file)
    
    
# main logic
def predict_laptop_price(brand,spec_rating,Ram,Ram_type,ROM,display_size,OS,
       warranty,screen_pixels,gpu_brand,gpu_tier,cpu_cores,cpu_threads,is_ssd,is_integrated):

    input_df = pd.DataFrame([[brand,spec_rating,Ram,Ram_type,ROM,display_size,OS,
       warranty,screen_pixels,gpu_brand,gpu_tier,cpu_cores,cpu_threads,is_ssd,is_integrated]],
    columns = ['brand', 'spec_rating', 'Ram', 'Ram_type', 'ROM', 'display_size', 'OS','warranty', 'screen_pixels', 'gpu_brand', 'gpu_tier','cpu_cores', 'cpu_threads', 'is_ssd', 'is_integrated']
    )
    
    #prediction
    prediction = model.predict(input_df)[0]
    return prediction
    
inputs = [

    gr.Dropdown(
        choices=["Apple","Acer", "MSI", "HP", "Asus", "Lenovo", "Dell"],
        label="Brand"
    ),

    gr.Slider(
        minimum=50,
        maximum=85,
        step=1,
        label="Specification Rating"
    ),

    gr.Dropdown(
        choices=["4GB", "8GB", "16GB", "32GB"],
        label="RAM"
    ),

    gr.Dropdown(
        choices=["DDR3","DDR4", "DDR5", "LPDDR5", "LPDDR4X"],
        label="RAM Type"
    ),

    gr.Dropdown(
        choices=["128GB", "256GB", "512GB", "1TB", "2TB"],
        label="Storage (ROM)"
    ),

    gr.Slider(
        minimum=13.0,
        maximum=16.5,
        step=0.1,
        label="Display Size (inches)"
    ),

    gr.Dropdown(
        choices=["Windows 11 OS", "Windows OS","DOS OS","Mac OS"],
        label="Operating System"
    ),

    gr.Radio(
        choices=[1, 2],
        label="Warranty (Years)"
    ),

    gr.Dropdown(
        choices=[2073600, 2304000, 3686400, 4096000, 4665600, 5184000],
        label="Screen Resolution (Pixels)"
    ),

    gr.Dropdown(
        choices=["Intel", "AMD", "NVIDIA","Apple"],
        label="GPU Brand"
    ),

    gr.Radio(
        choices=["Low", "Mid", "High"],
        label="GPU Tier"
    ),

    gr.Slider(
        minimum=2,
        maximum=24,
        step=2,
        label="CPU Cores"
    ),

    gr.Slider(
        minimum=4,
        maximum=32,
        step=2,
        label="CPU Threads"
    ),

    gr.Radio(
        choices=[0, 1],
        label="SSD Available (1 = Yes, 0 = No)"
    ),

    gr.Radio(
        choices=[0, 1],
        label="Integrated GPU (1 = Yes, 0 = No)"
    )

]
    
# interface
app = gr.Interface(
    fn=predict_laptop_price,
    inputs=inputs,
    outputs="text",
    title = "Laptop Price Prediction"
)

# launch

app.launch()