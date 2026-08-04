# Wind Resource Analysis Tool

A Streamlit app version of the wind measurement + ERA5 correlation / long-term
correction pipeline. No coding required to use it - the user maps their own
CSV columns interactively.

## Run locally

```bash
pip install -r requirements.txt
streamlit run wind_streamlit_app.py
```

Then open the local URL it prints (usually http://localhost:8501).

## Deploy for free (share with others) via Streamlit Community Cloud

1. Push `wind_streamlit_app.py` and `requirements.txt` to a GitHub repo.
2. Go to https://share.streamlit.io, sign in with GitHub.
3. Click "New app", point it at the repo/branch and `wind_streamlit_app.py`.
4. Deploy - you'll get a public URL you can send to anyone; they don't need
   Python installed.

## How to use

1. **Upload measurement CSV** - any column layout is fine.
2. **Map columns**: pick the timestamp column, date format, invalid-value
   codes (comma-separated, e.g. `9999, 999, -999`), and for each height you
   care about, the wind speed column (required) and wind direction column
   (optional - needed only for wind roses).
   - The app checks your date-format choice against the data and will warn
     you if it looks wrong (e.g. a huge chunk of rows failing to parse).
3. **Upload ERA5 CSV** - fixed columns: `Timestamp, Spd_100m_mps,
   Dir_100m_deg, Prs_0m_hPa, Tmp_2m_degC`.
4. **Set the UTC offset** of your measurement timestamps (ERA5 is assumed UTC).
5. Browse the result tabs: Data Availability, Monthly Means, Wind Rose,
   Shear Profile, Correlation, and Long-Term Result.
   - Shear is calculated using only heights above your chosen data-availability
     threshold (default 80%), fit once on the time-averaged profile (robust
     "profile method", not noisy per-timestamp averaging).
   - The long-term result uses orthogonal (total least-squares) regression
     against the full ERA5 record as the primary estimate, with the OLS result
     shown alongside for reference. The result at your chosen "height of
     interest" is obtained by shear-extrapolating from the measured height
     used in the correlation.
