# Wind Resource Analysis Tool

A Streamlit app with two separate tools, switchable from the sidebar:

1. **Long-Term Correction** - measurement + modelled data correlation and
   long-term wind speed pipeline. No coding required - the user maps their
   own CSV columns interactively.
2. **Measurement Campaign Planning** - preliminary wind resource look-up and
   LiDAR/FLiDAR siting: upload a site boundary, a turbine layout, and one or
   more modelled wind maps (ESRI ASCII grid `.asc`, e.g. Vortex map exports),
   click on an interactive map to place candidate measurement points, compare
   wind speed across points and maps, and run an automated K-Means-based
   search for the best measurement locations relative to your layout.

## Run locally

```bash
pip install -r requirements.txt
streamlit run wind_streamlit_app.py
```

Then open the local URL it prints (usually http://localhost:8501).

## Deploy for free (share with others) via Streamlit Community Cloud

1. Push `wind_streamlit_app.py`, `requirements.txt`, and the `.streamlit/`
   folder (containing `config.toml`) to a GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", point it at the repo/branch and `wind_streamlit_app.py`.
4. Deploy - you'll get a public URL you can send to anyone; they don't need
   Python installed.

Note: `geopandas` (used by the Measurement Campaign Planning mode) is a
heavier dependency than the rest of the app and may make the first Community
Cloud build take noticeably longer - this is normal and only happens once.

## How to use - Long-Term Correction

1. **Upload measurement CSV** - any column layout is fine.
2. **Map columns**: pick the timestamp column, date format, invalid-value
   codes (comma-separated, e.g. `9999, 999, -999`), and for each height you
   care about, the wind speed column (required) and wind direction column
   (optional - needed only for wind roses). Heights can be entered in any
   order - they're sorted low-to-high automatically everywhere in the results.
   - The app checks your date-format choice against the data and will warn
     you if it looks wrong (e.g. a huge chunk of rows failing to parse).
3. **Upload a modelled wind dataset** - any hourly modelled/reanalysis wind
   time series works (ERA5, CFSR, MERRA-2, Vortex, or similar). Vortex `.txt`
   exports are auto-detected and parsed automatically (timestamp, height, and
   timezone all read from the file header); other formats are mapped the
   same way as the measurement file (timestamp, wind speed, wind direction
   columns, plus the dataset's height).
4. **Set the timezone offsets** for both the measurement and modelled data
   (Vortex's offset is pre-filled from the file header; most reanalysis
   products are already UTC).
5. Browse the result tabs: Data Availability, Monthly Means, Wind Rose,
   Shear Profile, Correlation, and Long-Term Result.
   - Shear is calculated using only heights above your chosen data-availability
     threshold (default 80%), fit once on the time-averaged profile (robust
     "profile method", not noisy per-timestamp averaging).
   - Correlation is always done at the modelled dataset's height, using a
     direct measurement match if you mapped one close enough, or shear-
     extrapolating from your nearest mapped height otherwise. Sub-hourly
     modelled data is automatically averaged to hourly first.
   - The long-term result uses orthogonal (total least-squares) regression
     against the full modelled record as the primary estimate, with the OLS
     result shown alongside for reference. The result at your chosen "height
     of interest" is obtained by shear-extrapolating from the modelled
     dataset's height.
6. **Download**: package every chart (plus a summary and the availability
   table) into a single ZIP from the Download section at the bottom.

## How to use - Measurement Campaign Planning

1. **Site boundary**: upload a `.geojson` polygon of your site.
2. **Wind maps**: upload one or more `.asc` (ESRI ASCII grid) wind speed
   maps, each tagged with a source label and height - e.g. a Vortex map
   export at 100m and another at 150m.
3. **Turbine layout** (optional but needed for step 6): upload `.geojson`,
   `.gpkg`, or `.xlsx` (with Latitude/Longitude columns - column names are
   configurable). Raw `.shp` isn't supported directly since it's really
   several files bundled together - convert or export to one of the above.
4. **Interactive map**: the active wind map renders as a heatmap with the
   boundary and layout overlaid. Click anywhere inside the boundary to drop
   a labeled measurement point (A, B, C...); click "Fix measurement points"
   once you're happy with them.
5. **Compare wind speed at chosen points**: enter combinations like `A,B;A,C`
   to get an inline chart comparing wind speed at those points across every
   loaded wind map.
6. **Locate best measurement points**: choose how many measurement locations
   you want; the tool clusters your turbines with K-Means and searches inside
   the boundary for the point in each cluster that best represents that
   cluster's modelled wind speed (weighted against distance to the turbines),
   reporting mean/max deviation for each recommended point, downloadable as CSV.
