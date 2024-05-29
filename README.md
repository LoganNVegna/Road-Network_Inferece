# Road Network Inference
Please read the pdf


## ML_DT
Transportation network data collection project

NYC netwrok data is shared on Box (it is more than 25 MB). Look at the .DBF file and TOT_LANE for the number of lanes on each street. You can also map the file using the .shp file. https://cornell.box.com/s/novnxiavtl9pmth4p260p8x8s155limd

And here is the source for the street width: https://github.com/CityOfNewYork/nyc-geo-metadata/blob/master/Metadata/Metadata_StreetCenterline.md

### List of case studies:
Washington DC <br>
Baltimore <br>
Philadelphia <br>
New York City <br>
San Francisco <br>
San Jose <br>
Austin <br>
Denver <br>
Portland <br>
Seattle  <br>
Salt Lake City (Test Set)<br>

### Tentative timeline
02/20/22	research <br>
02/27/22	research <br>
03/06/22	dataset <br>
03/13/22	dataset <br>
03/20/22	build <br>
03/27/22	build/train <br>
04/03/22	train/test <br>
04/10/22	dataset <br>
04/17/22	build <br>
04/24/22	build/train <br>
05/01/22	train/test <br>
05/08/22	end of semester <br>

### Actual
02/20/22	research <br>
02/27/22	research <br>
03/06/22	explore codes, dataset, get paper code running <br>
03/13/22	explore codes, dataset, get paper code running <br>
03/20/22	create dataset, get paper code running <br>
03/27/22

## Training pipeline
1. Extract lane number and location from OSM and satellite Google Static Map
2. Process the data
3. Use pretrained ScRoadExtractor to get lane width and road mask
4. Train a model, likely CNN, to predict number of lanes
5. In the future, use lane width output of ScRoadExtractor as training data to train a model that predict both number of lanes and width

## Processing pipeline
Dont forget to extract the datasets before running the default code.

1. Use OSM to extract lane location 
2. For each city select 100 random points from the lane location and use Google Static Map to extract satellite image
3. Preprocess the data
4. Use ScRoadExtractor to get lane width and road mask
5. Use our trained model to get number of lanes
