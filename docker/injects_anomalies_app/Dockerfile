FROM python:3.7
MAINTAINER Réverton Luis Antunes Neundorf <revertonluis@gmail.com>

# ----------- To see the python stdout --------------------------
ENV PYTHONUNBUFFERED 1
# ---------------------------------------------------------------

# ---------- Create the directory of the app --------------------
RUN mkdir -p /app
# ---------------------------------------------------------------

# ---------- Set the working directory --------------------------
WORKDIR /app
# ---------------------------------------------------------------

# ------------ App config -----------------------------------------
# ENV FLASK_APP injects_anomalies.py
# ENV FLASK_RUN_HOST 0.0.0.0
# -----------------------------------------------------------------

# ------------- Copy the app into docker directory  ---------------
COPY . /app
# -----------------------------------------------------------------

# ------------- Install python dependencies -----------------------
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
# -----------------------------------------------------------------

# --------- Define the command to run the app ---------------------
CMD ["python", "injects_anomalies.py"]
# -----------------------------------------------------------------
