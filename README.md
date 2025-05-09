# Analysis of Partisan Voting in Michigan

This project builds a predictive pipeline for analyzing voter behavior using precinct-level election results, demographic features, and socioeconomic indicators. It integrates geospatial processing, feature engineering, and machine learning techniques to investigate electoral patterns in Michigan.

---

## Code and File Structure

The project is contained in a [git repository](https://github.com/nketchum/SI699) located on GitHub – clone and use the <code>main</code> branch.

<code>$ git clone https://github.com/nketchum/SI699</code>

Once you clone the repository, it will create the project directory and add all code files, including:

- <code>environment.yml</code> — Defines the execution environment.
- <code>\*.ipynb</code>, <code>\*.py</code>, <code>data/</code> dir, and <code>output</code> dir
- <code>OpenAI.key.disabled</code> — Placeholder file to securely store your OpenAI API key (if used).

---

## Environment Setup

The project uses Anaconda for environment and dependency management.

To create the environment:

- <code>$ conda env create -f environment.yml -n my-env-name</code>

Verify your setup:

- <code>$ python --version</code>
- <code>$ conda list</code>

---

## Data & Sources

The project uses a mix of federal, state, and non-profit data sources. See <code>data_directories.txt</code> in the project root directory to see how the data is structured. All data is considered public and owned by their respective providers.

- Note: the <code>custom_data</code> and <code>generated_data</code> directories are automatically added in the main project repository; there is no need to create that.

**OpenElections**

Provides [publicly-licensed precinct-level data for Michigan](https://github.com/openelections/openelections-data-mi) starting from the 2000 election cycle, which includes both general elections and primaries. [(Temp. download url)](https://www.dropbox.com/scl/fi/78ch4j24fa5fgnf89c9xm/openelections-data-mi.zip?rlkey=vnx06hb6baa3i6wwymqqu73hb&dl=0)

**Michigan Secretary of State**

- [State-managed historical election outcome data](https://miboecfr.nictusa.com/cgi-bin/cfr/precinct_srch.cgi) for years 2014-2022. [(Temp. download url)](https://www.dropbox.com/scl/fi/aqrc9zghy3g25gq9a5r81/elections-data-mi-sos.zip?rlkey=h2uvapatss3eiaeosvwzde8ar&dl=0)

  - Note: Access was suddenly restricted in mid-March, 2025.
  - A preserved copy of the data is temporarily available on [Dropbox](https://www.dropbox.com/scl/fi/aqrc9zghy3g25gq9a5r81/elections-data-mi-sos.zip?rlkey=h2uvapatss3eiaeosvwzde8ar&dl=0).
  - OpenElections data can mostly replace this data after processed via the <code>01b_election_data_openelections.ipynb</code> script.

- ArcGIS precinct boundary shapefiles provided by the Secretary of State from 2014 onward are available:

  - [2014 boundaries](https://gis-michigan.opendata.arcgis.com/datasets/Michigan::2014-voting-precincts)
  - [2016 boundaries](https://gis-michigan.opendata.arcgis.com/datasets/Michigan::2016-voting-precincts)
  - [2018 boundaries](https://gis-michigan.opendata.arcgis.com/datasets/Michigan::2018-voting-precincts)
  - [2020 boundaries](https://gis-michigan.opendata.arcgis.com/datasets/Michigan::2020-voting-precincts)
  - [2022 boundaries](https://gis-michigan.opendata.arcgis.com/datasets/Michigan::2022-voting-precincts)
  - [2024 boundaries](https://gis-michigan.opendata.arcgis.com/datasets/Michigan::2024-voting-precincts)

- Qualified Voter File

  - The [Qualified Voter File (QVF)](https://www.michigan.gov/sos/elections/admin-info/qvf) is also available upon request via FOIA, though it requires a fee.

**U.S. Census Bureau**

Provides high-resolution demographic and socioeconomic data via the American Community Survey (ACS) and decennial census. Yearly estimates supplement 10-year census intervals. "TigerLine" shapefiles are also available to help map precincts to certain geographic subdivisions.

- [Portal containing ACS datasets](https://data.census.gov/). [(Temp. download url)](https://www.dropbox.com/scl/fi/xuv33cksvcdu8mka33lnj/census.zip?rlkey=cn19tkxwfe38q16mpsbrdi473&dl=0)
- ["TigerLine" precinct shapefiles ](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html). [(Temp. download url)](https://www.dropbox.com/scl/fi/neenl5fl22sqkjfr493zb/voting_precincts.zip?rlkey=1hrsczreprhxbgstx3hp6958g&dl=0)

**FCC**

Federal data from the Federal Communications Commission (FCC) [containing keys](https://transition.fcc.gov/oet/info/maps/census/fips/fips.txt) that map counties from  Michigan-issued keys to federally-issued keys. [(Temp. download url)](https://www.dropbox.com/scl/fi/hs3hdah0ss51l51copsur/fcc.zip?rlkey=2z4pgwtkfkvszolmr17hf624c&dl=0)

**ODD**

 Open Data Delaware (ODD) offers a [repository of zip codes](https://github.com/OpenDataDE/State-zip-code-GeoJSON) and corresponding coordinates, used for mapping precincts to zip codes. [(Temp. download url)](https://www.dropbox.com/scl/fi/zesj2c6t4tkvde510u3rl/OpenDataDE.zip?rlkey=0kln40inh0mya2g0w45ktyvr4&dl=0)

 **Data.gov**

 School district [boundary data](https://catalog.data.gov/dataset/school-district-characteristics-2020-21-99af4/resource/359fcb51-e4da-4118-8868-5b87046ba7b3?inner_span=True) is available in Data.gov's data catalog. [(Temp. download url)](https://www.dropbox.com/scl/fi/gn3wiyx6dvnultn71r29f/school_districts.zip?rlkey=del6q0oxtvkcthly6gvy9vzj4&dl=0)

 **NPMS**

 The National Pipeline Mapping System (NPMS) offers [boundary data](https://catalog.data.gov/dataset/school-district-characteristics-2020-21-99af4/resource/359fcb51-e4da-4118-8868-5b87046ba7b3?inner_span=True) on high-population areas, which help clustering voting regions. [(Temp. download url)](https://www.dropbox.com/scl/fi/cv2m0l9o8h4nolzvopb1u/USDOT.zip?rlkey=82flumcm1quacfx1d5y0mf496&dl=0)

---

## ChatGPT API Integration (Optional)

If the LLM flag is enabled in <code>11_analysis.py</code>, a ChatGPT API key is required.

1. Request an API key from: https://platform.openai.com/account/api-keys
2. Save the key in a file named <code>OpenAI.key</code> (one line only).
3. Do not commit this file — it contains sensitive credentials.

---

## Usage

**Run the full pipeline**:

1. Navigate to the project root in your terminal.
2. Activate the Conda environment.
3. Execute the main script:

  - <code>$ cd /path/to/project/dir/</code>
  - <code>$ conda activate</code>
  - <code>$ python \_main.py</code>

The terminal will display sequential outputs as scripts execute — including data preprocessing, feature generation, model fitting, and plotting.

Note: Prediction targets and feature selections must currently be changed directly in the source code. A Flask-based web interface is planned to allow interactive control of these parameters in the future.

**Alternative execution**

If memory or compute power is an issue, you may also execute each script within an iPython notebook in sequence as indicated in the <code>\_main.py</code> file. In this case, begin execution at <code>00_election_data_sos.ipynb</code> and continue on to <code>01_election_data_sos.ipynb</code>, <code>01b_election_data_openelections.ipynb</code>, <code>02_vote_changes.ipynb</code> and so on until all <code>\*.ipynb</code> scripts have been executed.

**Outputs**

All outputs are located in the <code>outputs/</code> directory, located in the root project directory. Within are directories for maps, personas, reports, and other items that were used to generate the class-submitted project report.

---

## Project Status

This repository contains the **foundational** logic for a larger application.

Upcoming development includes:

- Refactoring into modular, object-oriented components
- Adherence to PEP 8 and professional software engineering practices
- Support for additional prediction targets and modeling strategies
- Development of a web-based user interface for configuration and visualization

---

## Authors and acknowledgment

All original Python scripts and project architecture were developed by Nicholas Ketchum. All other attributions appear inline with code files as comments.

---

## Disclaimer

This project was conceived, developed, and submitted as a mastery project for the University of Michigan School of Information. No warranties, guarantees, or other claims are attached to this project. This project is for academic uses only, and may contain inaccurate or incomplete information.