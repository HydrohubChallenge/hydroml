import pandas as pd
import plotly.express as px
import dash_table
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from dataclasses import dataclass, field
from typing import List
import random
import numpy
from dash_extensions import Download
from dash_extensions.snippets import send_data_frame
import base64
import io
import more_itertools
import plotly.graph_objects as go


def is_number(a):
    if not a:
        return False
    try:
        float(a)
        return True
    except ValueError:
        return False


# ------------------------ Anomalies Class -----------------------
@dataclass
class Anomalies:
    
    # The dataframe
    data: pd.DataFrame = None
    
    # indexes from previous simulation
    siv_ps: List = field(default_factory=lambda: [])
    ds_ps: List = field(default_factory=lambda: [])
    sv_ps: List = field(default_factory=lambda: [])
        
    # Persistence min and max
    persistence_min: int = 10
    persistence_max: int = 10
    
    # Amplitude min and max
    amplitude_min: float = 1.
    amplitude_max: float = 1.
    
    # Minimum distance between anomalies
    anomalies_dist_min: int = 5
    
    def save_data_with_anomalies(self, csv_name="anomalies.csv"):
        """ Method that saves the dataframe with anomalies """
        self.data.to_csv(path_or_buf=csv_name, index=False)
            
    def add_anomalies_name_and_value_columns(self):
        """ Method that add anomalies columns name and values """
        self.data["anomaly_name"] = ""
        self.data["anomaly_value"] = ""
        
    def choose_index(self, persistence=0):
        
        # Find all the allowed indexes
        allowed_indexes = list(numpy.where(self.data["anomaly_value"] == "")[0])

        # Find the indexes that must be removed
        indexes_to_remove = []
        for ai in more_itertools.consecutive_groups(allowed_indexes):
            ai_list = list(ai)
            if ai_list:
                # Remove points before
                indexes_to_remove.extend(list(range(ai_list[0], ai_list[0] + self.anomalies_dist_min)))

                # Remove points after
                indexes_to_remove.extend(
                    list(range(ai_list[-1], ai_list[-1] - self.anomalies_dist_min - persistence, -1))
                )
        
        # Remove the anomalies_dist_min and persistence points
        allowed_indexes = list(set(allowed_indexes) - set(indexes_to_remove))
        return random.choice(allowed_indexes)

    def spikes_in_values(
        self, 
        amount='',
        anomaly_name='',
        persistence_min='', 
        persistence_max='',
        amplitude_min='',
        amplitude_max=''        
    ):
        
        # ------- Remove the previous anomaly simulation --------
        # get the indexes from the previous simulation
        if self.siv_ps:
            self.data.loc[self.siv_ps, "anomaly_name"] = ""
            self.data.loc[self.siv_ps, "anomaly_value"] = ""
        # -------------------------------------------------------    
        
        # Clean self.siv_ps for a new simulation
        self.siv_ps = []
        
        # Check if the arguments are numeric
        if is_number(amount):
            for _ in range(int(amount)):
                
                amin = self.amplitude_min
                amax = self.amplitude_max
                
                if is_number(amplitude_min):
                    amin = float(amplitude_min)
                
                if is_number(amplitude_max):
                    amax = float(amplitude_max)
                
                amplitude = random.uniform(min([amin, amax]), max([amin, amax]))              
               
                # Choose a proper index for the anomaly
                spike_index = self.choose_index()
                value = self.data.at[spike_index, settings["df_y_column"]] + amplitude
                self.data.at[spike_index, "anomaly_name"] = anomaly_name
                self.data.at[spike_index, "anomaly_value"] = value
                self.siv_ps.append(spike_index)
                
    def stationary_values(
        self, 
        amount='',
        anomaly_name='',
        persistence_min='', 
        persistence_max='',
        amplitude_min='',
        amplitude_max=''   
    ):
        
        # ------- Remove the previous anomaly simulation --------
        # get the indexes from the previous simulation
        if self.sv_ps:
            self.data.loc[self.sv_ps, "anomaly_name"] = ""
            self.data.loc[self.sv_ps, "anomaly_value"] = ""
        # -------------------------------------------------------    
        
        # Clean self.sv_ps for a new simulation
        self.sv_ps = []
        # Check if the arguments are numeric
        if is_number(amount):
            
            # ---------- Persistence -------------------
            pmin = self.persistence_min
            pmax = self.persistence_max
                
            if is_number(persistence_min):
                pmin = int(persistence_min)
                
            if is_number(persistence_max):
                pmax = int(persistence_max)
            # ------------------------------------------
            
            for _ in range(int(amount)):
                
                # Always a random persistence for each anomaly
                persistence = random.randint(min([pmin, pmax]), max([pmin, pmax]))                                
                                              
                # Choose a proper index for the anomaly
                index_s = self.choose_index(persistence=persistence)
                index_e = index_s + persistence
                
                self.data.loc[index_s:index_e, "anomaly_name"] = anomaly_name
                self.data.loc[index_s:index_e, "anomaly_value"] = self.data.at[index_s, settings["df_y_column"]]
                self.sv_ps.extend(list(range(index_s, index_e + 1)))
                    
    def sensor_displacement(
        self, 
        amount='',
        anomaly_name='',
        persistence_min='', 
        persistence_max='',
        amplitude_min='',
        amplitude_max=''
    ):
        
        # ------- Remove the previous anomaly simulation --------
        # get the indexes from the previous simulation
        if self.ds_ps:
            self.data.loc[self.ds_ps, "anomaly_name"] = ""
            self.data.loc[self.ds_ps, "anomaly_value"] = ""
        # -------------------------------------------------------    
        
        # Clean self.ds_ps for a new simulation
        self.ds_ps = []
        # Check if the arguments are numeric
        if amount.isnumeric():
            
            # ---------- Amplitude -------------------
            amin = self.amplitude_min
            amax = self.amplitude_max
                
            if is_number(amplitude_min):
                amin = float(amplitude_min)
                
            if is_number(amplitude_max):
                amax = float(amplitude_max)                
            # ------------------------------------------
            
            # ---------- Persistence -------------------
            pmin = self.persistence_min
            pmax = self.persistence_max
                
            if is_number(persistence_min):
                pmin = int(persistence_min)
                
            if is_number(persistence_max):
                pmax = int(persistence_max)                
            # ------------------------------------------
            
            for _ in range(int(amount)):
                
                # Always a random amplitude and persistence for each anomaly
                amplitude = random.uniform(min([amin, amax]), max([amin, amax]))
                persistence = random.randint(min([pmin, pmax]), max([pmin, pmax]))
                                
                # Choose a proper index for the anomaly
                index_s = self.choose_index(persistence=persistence)
                index_e = index_s + persistence
                
                self.data.loc[index_s:index_e, "anomaly_name"] = anomaly_name
                self.data.loc[index_s:index_e, "anomaly_value"] = self.data.loc[
                    index_s:index_e, settings["df_y_column"]
                ] + amplitude
                self.ds_ps.extend(list(range(index_s, index_e + 1)))


def decode_csv_content(csv_content=None, filename=None):
    df = None
    if csv_content:
        content_type, content_string = csv_content.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(
                    io.StringIO(decoded.decode('utf-8')),
                    float_precision='round_trip'
                )
            elif 'xls' in filename:
                # Assume that the user uploaded an excel file
                df = pd.read_excel(io.BytesIO(decoded))
        except Exception as e:
            print(e)
            
    return df


# ---------- The project Settings --------------
settings = {
    "df_x_column": "datetime",
    "df_y_column": "measured",
    "plot_settings": {
        "x_label": "Date",
        "y_label": "Water Level",        
        "Original_Values": {
            "color": "blue"
        }
    }
}

# ------------------ The anomlies and their methods -------------------
anomalies_methods = {
    "Spikes": "spikes_in_values",
    "Stationary Values": "stationary_values",
    "Sensor Displacement": "sensor_displacement" 
}

# Update the settings_plot from settings with the anomlies colors
colors = ["black", "red", "green", "black"]
for anomaly, index in zip(anomalies_methods, range(len(list(anomalies_methods.keys())))):
    settings["plot_settings"].update(
        {
            anomaly: {
                "color": colors[index]
            }
        }
    )


# -------------------- params --------------------------
reference_parameters = {
    "load_csv_n_clicks": 0,
    "injects_anomalies_n_clicks": 0,
    "upload_dataframe_content": "",
    "fig": px.scatter(),
    "plots_first_index": {}
}


# ----------------------- Start Anomalies Class and add the dataframe ---------------
anomalies = Anomalies()

# -------------------------- Tables ------------------------------
anomalies_table = dash_table.DataTable(    
    id='anomalies-table',
    columns=(
        [
            {
                'id': 'Anomaly', 'name': "Anomaly", 'editable': False
            },
            {
                'id': 'Amount', 'name': "Amount", 'editable': True
            },
            {
                'id': 'Amplitude (Min)', 'name': "Amplitude (Min)", 'editable': True
            },
            {
                'id': 'Amplitude (Max)', 'name': "Amplitude (Max)", 'editable': True
            },
            {
                'id': 'Persistence (Min)', 'name': "Persistence (Min)", 'editable': True
            },
            {
                'id': 'Persistence (Max)', 'name': "Persistence (Max)", 'editable': True
            }
            
        ]
    ),
    data=[
        {
            "Anomaly": anomaly_name,
            "Amount": "",
            "Persistence (Min)": "", 
            "Persistence (Max)": "",
            "Amplitude (Min)": "",
            "Amplitude (Max)": ""
        }
        for anomaly_name in anomalies_methods
    ]
)

fig_table = dash_table.DataTable(
    
    id='fig-table',
    columns=(
        [
            {
                'id': 'Date', 'name': "Date", 'editable': False
            },
            {
                'id': 'Original Value', 'name': "Original Value", 'editable': False,
            },
            {
                'id': 'Anomaly', 'name': "Anomaly", 'editable': False,
            },
            {
                'id': 'Anomaly Value', 'name': "Anomaly Value", 'editable': False
            }
        ]
    ),
    data=[]
)


# ------------------- App --------------------------
app = dash.Dash(__name__)

# ------------------- App layout -------------------
app.layout = html.Div([
    anomalies_table,
    dcc.Upload(
        id='upload-dataframe',
        children=html.Div(
            [
                html.Button('Load csv', id='load-dataframe-button', n_clicks=0)
            ]
        )
    ),
    html.Button('Injects Anomalies', id='injects-anomalies-button', n_clicks=0),
    html.Button('Download csv with Anomalies', id='download-dataframe-with-anomalies-button', n_clicks=0),
    dcc.Graph(
        id='anomalies-fig', 
        figure=reference_parameters['fig'],
    ),
    fig_table,
    Download(id="download-anomalies-csv"),    
    html.Div(id='output-data-upload')
])


# ---------------------------- Select Data display table Callback -----------------------
@app.callback(
    Output('fig-table', 'data'),
    [
        Input('anomalies-fig', 'selectedData')        
    ]
)
def select_data_display_table(selected_data):
    data = []
    if selected_data:
        for point in selected_data['points']:
            pi = point['pointIndex']
            cn = point['curveNumber']
            correct_index = pi + reference_parameters["plots_first_index"][cn]
            data.append(
                {
                    "Date": anomalies.data.at[correct_index, settings["df_x_column"]],
                    "Original Value": anomalies.data.at[correct_index, settings["df_y_column"]],
                    "Anomaly": anomalies.data.at[correct_index, "anomaly_name"], 
                    "Anomaly Value": anomalies.data.at[correct_index, "anomaly_value"]
                }
            )
            
    return data


# ---------------------------- Download Csv with Anomalies  Callback -----------------------
@app.callback(
    Output("download-anomalies-csv", "data"),
    [
        Input('download-dataframe-with-anomalies-button', 'n_clicks')
    ]
)
def download_dataframe_with_anomalies(n_clicks):
    if n_clicks:
        return send_data_frame(anomalies.data.to_csv, filename="anomalies.csv")
    else:
        return None


# -------------------------- Load CSV and Injects Anomalies Callback ----------------------
@app.callback(
    Output('anomalies-fig', 'figure'),
    [
        Input('load-dataframe-button', 'n_clicks'),
        Input('injects-anomalies-button', 'n_clicks'),
        Input('anomalies-table', 'data'),
        Input('upload-dataframe', 'contents')
    ],
    [
        State('upload-dataframe', 'filename')
    ]   
)
def load_csv_and_injects_anomalies(
    load_csv_n_clicks,
    injects_anomalies_n_clicks,
    anomalies_table_data,
    upload_dataframe_content,
    upload_dataframe_filename
):
      
    # ----------------------------- LOAD THE CSV ------------------------------------
    if load_csv_n_clicks != reference_parameters["load_csv_n_clicks"]:
        if upload_dataframe_content:
            if upload_dataframe_content != reference_parameters["upload_dataframe_content"]:
            
                # Load and decode the csv
                df = decode_csv_content(csv_content=upload_dataframe_content, filename=upload_dataframe_filename)
                anomalies.data = df.copy()
                anomalies.add_anomalies_name_and_value_columns()

                # Create a figure for the csv
                fig = px.scatter(df, x=settings["df_x_column"], y=settings["df_y_column"], render_mode='webgl')
                fig.data[0].update(mode='markers+lines', marker={'size': 1, 'color': 'blue'})
                fig.update_layout(
                    clickmode='event+select',
                    yaxis={"title": settings["plot_settings"]["y_label"]},
                    xaxis={"title": settings["plot_settings"]["x_label"]}
                )

                # Saving the first index of the plot because each plot
                # will restart with index = 0
                reference_parameters["plots_first_index"][0] = 0
                
                # Update Reference Parameters    
                reference_parameters["load_csv_n_clicks"] = load_csv_n_clicks
                reference_parameters["upload_dataframe_content"] = upload_dataframe_content
                reference_parameters["fig"] = fig
    
    # ------------------------ INJECTS ANOMALIES -----------------------------------------
    if injects_anomalies_n_clicks != reference_parameters["injects_anomalies_n_clicks"]:
        if upload_dataframe_content:
            
            # Injects anomalies in the anomlies.data and return the
            for aft in anomalies_table_data:

                getattr(anomalies, anomalies_methods[aft["Anomaly"]])(
                    amount=aft["Amount"],
                    anomaly_name=aft["Anomaly"],
                    persistence_min=aft["Persistence (Min)"],
                    persistence_max=aft["Persistence (Max)"],
                    amplitude_min=aft["Amplitude (Min)"],
                    amplitude_max=aft["Amplitude (Max)"]
                )

            # ------------------ Break the fig in various Subplots ------------------------------
            # Get the indexes for original values (without anomalies) and indexes with anomalies
            original_indexes = numpy.where(anomalies.data["anomaly_value"] == "")[0]
            anomalies_indexes = numpy.where(anomalies.data["anomaly_value"] != "")[0]

            # The indexes with each plot
            plots_indexes = []

            # Break the indexes for each plot with original values
            for plot in more_itertools.consecutive_groups(original_indexes):
                plots_indexes.append(list(plot))

            # Break the indexes for each plot with anomalies
            for plot in more_itertools.consecutive_groups(anomalies_indexes):
                plots_indexes.append(list(plot))

            # Define a fig
            # render_mode MUST BE webgl
            fig = px.scatter(render_mode='webgl')

            # Create a subplot for each plot_indexes
            for plot_indexes, plot_id in zip(plots_indexes, range(len(plots_indexes))):

                # Add the subplots with
                # Get the name of the anomaly
                anomaly_name = anomalies.data.loc[plot_indexes[0], "anomaly_name"]
                y_var = "anomaly_value"
                if not anomaly_name:
                    anomaly_name = "Original_Values"
                    y_var = settings["df_y_column"]

                # Get x and y
                plot_x = anomalies.data.loc[plot_indexes, settings["df_x_column"]].tolist()
                plot_y = anomalies.data.loc[plot_indexes, y_var].tolist()

                # Saving the first index of the plot because each plot
                # will restart with index = 0
                reference_parameters["plots_first_index"][plot_id] = plot_indexes[0]

                # To connect the plots add 1 point before the first point of the plot
                # and 1 point after the last point of the plot
                if anomaly_name != "Original_Values":

                    # 1 point before the first point of the plot
                    if plot_indexes[0] > 0:
                        plot_x.insert(0, anomalies.data.at[plot_indexes[0] - 1, settings["df_x_column"]])
                        plot_y.insert(0, anomalies.data.at[plot_indexes[0] - 1, settings["df_y_column"]])

                        # Fix in case of anomaly
                        reference_parameters["plots_first_index"][plot_id] = plot_indexes[0] - 1

                    # 1 point after the last point of the plot
                    if plot_indexes[-1] < anomalies.data.shape[0]:
                        plot_x.append(anomalies.data.at[plot_indexes[-1] + 1, settings["df_x_column"]])
                        plot_y.append(anomalies.data.at[plot_indexes[-1] + 1, settings["df_y_column"]])

                fig.add_traces(
                    # ScatterGL for performance
                    go.Scattergl(
                        x=plot_x, y=plot_y,
                        mode='markers+lines',
                        marker={'size': 1, 'color': settings["plot_settings"][anomaly_name]["color"]},
                        line={'color': settings["plot_settings"][anomaly_name]["color"]}
                    )
                )
            fig.update_layout(
                clickmode='event+select',
                showlegend=False,
                yaxis={"title": settings["plot_settings"]["y_label"]},
                xaxis={"title": settings["plot_settings"]["x_label"]}
            )
            # -----------------------------------------------------------------------

            # Update Reference Parameters
            reference_parameters["injects_anomalies_n_clicks"] = injects_anomalies_n_clicks
            reference_parameters["upload_dataframe_content"] = upload_dataframe_content
            reference_parameters["fig"] = fig

    return reference_parameters["fig"]


# --------------------- MAIN --------------------
if __name__ == '__main__':
    app.run_server()#host="0.0.0.0")
