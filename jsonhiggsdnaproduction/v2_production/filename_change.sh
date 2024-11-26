#!/bin/bash

# Check if a path argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <base_path>"
    exit 1
fi

base_path="$1"

# List of directories
directories=(
"NMSSM_X300_Y60" "NMSSM_X300_Y70" "NMSSM_X300_Y80" "NMSSM_X300_Y90" "NMSSM_X300_Y95" "NMSSM_X300_Y100" "NMSSM_X300_Y125" "NMSSM_X300_Y150"
"NMSSM_X400_Y60" "NMSSM_X400_Y70" "NMSSM_X400_Y80" "NMSSM_X400_Y90" "NMSSM_X400_Y95" "NMSSM_X400_Y100" "NMSSM_X400_Y125" "NMSSM_X400_Y150" "NMSSM_X400_Y200"
"NMSSM_X500_Y60" "NMSSM_X500_Y70" "NMSSM_X500_Y80" "NMSSM_X500_Y90" "NMSSM_X500_Y95" "NMSSM_X500_Y100" "NMSSM_X500_Y125" "NMSSM_X500_Y150" "NMSSM_X500_Y200" "NMSSM_X500_Y300"
"NMSSM_X550_Y60" "NMSSM_X550_Y70" "NMSSM_X550_Y80" "NMSSM_X550_Y90" "NMSSM_X550_Y95" "NMSSM_X550_Y100" "NMSSM_X550_Y125" "NMSSM_X550_Y150" "NMSSM_X550_Y200" "NMSSM_X550_Y300" "NMSSM_X550_Y400"
"NMSSM_X600_Y60" "NMSSM_X600_Y70" "NMSSM_X600_Y80" "NMSSM_X600_Y90" "NMSSM_X600_Y95" "NMSSM_X600_Y100" "NMSSM_X600_Y125" "NMSSM_X600_Y150" "NMSSM_X600_Y200" "NMSSM_X600_Y300" "NMSSM_X600_Y400"
"NMSSM_X650_Y60" "NMSSM_X650_Y70" "NMSSM_X650_Y80" "NMSSM_X650_Y90" "NMSSM_X650_Y100" "NMSSM_X650_Y125" "NMSSM_X650_Y150" "NMSSM_X650_Y200" "NMSSM_X650_Y300" "NMSSM_X650_Y400" "NMSSM_X650_Y500"
"NMSSM_X700_Y60" "NMSSM_X700_Y70" "NMSSM_X700_Y80" "NMSSM_X700_Y90" "NMSSM_X700_Y100" "NMSSM_X700_Y125" "NMSSM_X700_Y150" "NMSSM_X700_Y200" "NMSSM_X700_Y300" "NMSSM_X700_Y400" "NMSSM_X700_Y500"
"NMSSM_X800_Y60" "NMSSM_X800_Y70" "NMSSM_X800_Y80" "NMSSM_X800_Y90" "NMSSM_X800_Y95" "NMSSM_X800_Y100" "NMSSM_X800_Y125" "NMSSM_X800_Y150" "NMSSM_X800_Y200" "NMSSM_X800_Y300" "NMSSM_X800_Y400" "NMSSM_X800_Y500" "NMSSM_X800_Y600"
"NMSSM_X900_Y60" "NMSSM_X900_Y70" "NMSSM_X900_Y80" "NMSSM_X900_Y90" "NMSSM_X900_Y95" "NMSSM_X900_Y100" "NMSSM_X900_Y125" "NMSSM_X900_Y150" "NMSSM_X900_Y200" "NMSSM_X900_Y300" "NMSSM_X900_Y400" "NMSSM_X900_Y500" "NMSSM_X900_Y600"
"NMSSM_X1000_Y60" "NMSSM_X1000_Y70" "NMSSM_X1000_Y80" "NMSSM_X1000_Y90" "NMSSM_X1000_Y95" "NMSSM_X1000_Y100" "NMSSM_X1000_Y125" "NMSSM_X1000_Y150" "NMSSM_X1000_Y200" "NMSSM_X1000_Y300" "NMSSM_X1000_Y400" "NMSSM_X1000_Y500" "NMSSM_X1000_Y600"
"NMSSM_X1200_Y60" "NMSSM_X1200_Y70" "NMSSM_X1200_Y80" "NMSSM_X1200_Y90" "NMSSM_X1200_Y95" "NMSSM_X1200_Y100" "NMSSM_X1200_Y125" "NMSSM_X1200_Y150" "NMSSM_X1200_Y200" "NMSSM_X1200_Y400" "NMSSM_X1200_Y600" "NMSSM_X1200_Y800" "NMSSM_X1200_Y1000"
"NMSSM_X1400_Y60" "NMSSM_X1400_Y70" "NMSSM_X1400_Y80" "NMSSM_X1400_Y90" "NMSSM_X1400_Y95" "NMSSM_X1400_Y100" "NMSSM_X1400_Y125" "NMSSM_X1400_Y150" "NMSSM_X1400_Y200" "NMSSM_X1400_Y300" "NMSSM_X1400_Y400" "NMSSM_X1400_Y500" "NMSSM_X1400_Y600" "NMSSM_X1400_Y800" "NMSSM_X1400_Y1000" "NMSSM_X1400_Y1200"
"NMSSM_X1600_Y60" "NMSSM_X1600_Y70" "NMSSM_X1600_Y80" "NMSSM_X1600_Y90" "NMSSM_X1600_Y95" "NMSSM_X1600_Y100" "NMSSM_X1600_Y125" "NMSSM_X1600_Y150" "NMSSM_X1600_Y200" "NMSSM_X1600_Y300" "NMSSM_X1600_Y400" "NMSSM_X1600_Y600" "NMSSM_X1600_Y800" "NMSSM_X1600_Y1000" "NMSSM_X1600_Y1200" "NMSSM_X1600_Y1400"
"NMSSM_X1800_Y60" "NMSSM_X1800_Y70" "NMSSM_X1800_Y80" "NMSSM_X1800_Y90" "NMSSM_X1800_Y95" "NMSSM_X1800_Y100" "NMSSM_X1800_Y125" "NMSSM_X1800_Y150" "NMSSM_X1800_Y300" "NMSSM_X1800_Y400" "NMSSM_X1800_Y500" "NMSSM_X1800_Y600" "NMSSM_X1800_Y800" "NMSSM_X1800_Y1000" "NMSSM_X1800_Y1200" "NMSSM_X1800_Y1400" "NMSSM_X1800_Y1600"
"NMSSM_X2000_Y60" "NMSSM_X2000_Y70" "NMSSM_X2000_Y80" "NMSSM_X2000_Y90" "NMSSM_X2000_Y95" "NMSSM_X2000_Y100" "NMSSM_X2000_Y125" "NMSSM_X2000_Y150" "NMSSM_X2000_Y200" "NMSSM_X2000_Y300" "NMSSM_X2000_Y400" "NMSSM_X2000_Y500" "NMSSM_X2000_Y600" "NMSSM_X2000_Y800" "NMSSM_X2000_Y1000" "NMSSM_X2000_Y1200" "NMSSM_X2000_Y1400" "NMSSM_X2000_Y1600" "NMSSM_X2000_Y1800"
"NMSSM_X2500_Y60" "NMSSM_X2500_Y70" "NMSSM_X2500_Y80" "NMSSM_X2500_Y90" "NMSSM_X2500_Y95" "NMSSM_X2500_Y100" "NMSSM_X2500_Y125" "NMSSM_X2500_Y150" "NMSSM_X2500_Y200" "NMSSM_X2500_Y300" "NMSSM_X2500_Y400" "NMSSM_X2500_Y500" "NMSSM_X2500_Y600" "NMSSM_X2500_Y800" "NMSSM_X2500_Y1000" "NMSSM_X2500_Y1200" "NMSSM_X2500_Y1400" "NMSSM_X2500_Y1600" "NMSSM_X2500_Y1800" "NMSSM_X2500_Y2000"
"NMSSM_X3000_Y60" "NMSSM_X3000_Y70" "NMSSM_X3000_Y80" "NMSSM_X3000_Y90" "NMSSM_X3000_Y95" "NMSSM_X3000_Y100" "NMSSM_X3000_Y125" "NMSSM_X3000_Y150" "NMSSM_X3000_Y200" "NMSSM_X3000_Y300" "NMSSM_X3000_Y400" "NMSSM_X3000_Y500" "NMSSM_X3000_Y600" "NMSSM_X3000_Y800" "NMSSM_X3000_Y1000" "NMSSM_X3000_Y1200" "NMSSM_X3000_Y1400" "NMSSM_X3000_Y1600" "NMSSM_X3000_Y1800" "NMSSM_X3000_Y2000" "NMSSM_X3000_Y2600"
"NMSSM_X3500_Y60" "NMSSM_X3500_Y70" "NMSSM_X3500_Y80" "NMSSM_X3500_Y90" "NMSSM_X3500_Y100" "NMSSM_X3500_Y125" "NMSSM_X3500_Y150" "NMSSM_X3500_Y200" "NMSSM_X3500_Y300" "NMSSM_X3500_Y400" "NMSSM_X3500_Y500" "NMSSM_X3500_Y600" "NMSSM_X3500_Y800" "NMSSM_X3500_Y1200" "NMSSM_X3500_Y1400" "NMSSM_X3500_Y1600" "NMSSM_X3500_Y1800" "NMSSM_X3500_Y2000" "NMSSM_X3500_Y2600" "NMSSM_X3500_Y3000"
"NMSSM_X4000_Y70" "NMSSM_X4000_Y80" "NMSSM_X4000_Y90" "NMSSM_X4000_Y95" "NMSSM_X4000_Y100" "NMSSM_X4000_Y125" "NMSSM_X4000_Y150" "NMSSM_X4000_Y200" "NMSSM_X4000_Y300" "NMSSM_X4000_Y400" "NMSSM_X4000_Y500" "NMSSM_X4000_Y600" "NMSSM_X4000_Y800" "NMSSM_X4000_Y1000" "NMSSM_X4000_Y1200" "NMSSM_X4000_Y1400" "NMSSM_X4000_Y1600" "NMSSM_X4000_Y1800" "NMSSM_X4000_Y2000" "NMSSM_X4000_Y2600" "NMSSM_X4000_Y3000" "NMSSM_X4000_Y3500"
)

# Function to rename files
rename_files_in_nominal() {
    for dir in "${directories[@]}"; do
        nominal_path="${base_path}/${dir}/nominal"
        if [ -d "$nominal_path" ]; then
            for file in "$nominal_path"/*; do
                if [ -f "$file" ]; then
                    filename=$(basename -- "$file")
                    if [ "$filename" == "NOTAG_merged.parquet" ] || [ "$filename" == "NOTAG_NOTAG_merged.parquet" ]; then
                        new_file_path="${nominal_path}/NOTAG_NOTAG_merged.parquet"
                        mv "$file" "$new_file_path"
                        echo "Renamed $file to $new_file_path"
                    fi
                fi
            done
        else
            echo "Directory $nominal_path does not exist"
        fi
    done
}

# Execute the function
rename_files_in_nominal

