
# Step 1
for year in {1993..2026}; do
wget "https://s3-ap-southeast-2.amazonaws.com/silo-open-data/Official/annual/monthly_rain/${year}.monthly_rain.nc"
# Potentially hang (change year above and uncomment below)
# sec=$(shuf -i 7-11 -n 1)
# sleep $sec
done

# Step 2
cdo mergetime ????.monthly_rain.nc silo_monthly_rain.nc

# Step 3
ncks -d lat,-25,-22 -d lon,142,145 silo_monthly_rain.nc -O silo_monthly_rain_iceds.nc

echo "Done"