# brain_region_nomograms
This project creates nomograms for different brain regions for participants of ages 44-82. The nomograms are adjusted for genetic predispositions.
There are currently two brain regions this project works for, the amygdala volume and the hippocampal volume. To add another brain region,
create an enum that follows the same structure as `AmygdalaVolume`, and add the UKB column names for that region in `ukb_table_column_names.py`, 
following the format:
  ```
  l_{brain_region}_vol_cn = ...
  r_{brain_region}_vol_cn = ...
  ```
  
 Then, to run the analysis for that brain region, run the main file. In order to run it you need access to the needed datasets.
 The main file will
  1. Get all the datasets needed and merge them.
  2. Preprocess the merged dataset.  <br />
    a. This includes removing all participants with any neurological or psychiatric disorders, head trauma,
substance abuse, or cardiovascular disorders. <br />
    b. Correcting for head size and scan date. <br />
    c. Removing participants whose head size is 5 mean absolute errors away from the median. <br />
    d. Only including participants of ages 44-82 of white British descent. <br />
  3. Run the SWA analysis stratified by sex and hemisphere.
  4. Run the GPR analysis stratified by sex and hemisphere.
  (The above step takes a long time, so its resulting data is saved and can now be retrieved by the next bit of code.)
  5. Plot all desired nomograms 
  6. Do PGS score, get top and bottom 30% and perform GPR analysis stratified by sex and hemisphere.
  6. Calculate the difference between all genetically adjusted and unadjusted nomograms and perform a t-test to see if difference is significant.
  
  All nomograms are saved in the saved_nomograms folder, and all data in the saved_data folder. 
    
  
