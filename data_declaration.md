## Data Declaration

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

 Note: this content is also included in the README.md file.
 