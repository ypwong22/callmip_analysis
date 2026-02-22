module load nco

python makepointdata.py --site DK-Sor --sitegroup CalLMIP --ccsm_input ${WORLD}/e3sm/inputdata/ --mysimyr 1850

# edit the PCT_SAND & PCT_CLAY
ncap2 -s 'PCT_SAND(0)=60.0f; PCT_SAND(1)=60.0f; PCT_SAND(2)=63.0f; PCT_SAND(3)=63.0f; PCT_CLAY(0)=13.0f; PCT_CLAY(1)=13.0f; PCT_CLAY(2)=12.0f; PCT_CLAY(3)=13.0f' ./temp/surfdata.nc ./temp/surfdata_obs.nc

mv ./temp ${WORLD}/e3sm/inputdata/CalLMIP/1x1pt_DK-Sor