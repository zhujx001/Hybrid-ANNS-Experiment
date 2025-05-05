#!/usr/bin/env python3

import subprocess
import os

def main():
    # Display startup reminder
    print("=============================================================")
    print("IMPORTANT: Please ensure the dataset paths are correct!")
    print("Check that the following directories exist and contain data:")
    print("  - ../../../data/Experiment/rangefilterData/datasets/")
    print("  - ../../../data/Experiment/temp/DSG/")
    print("")
    print("Tip: You can also run the shell scripts individually to experiment:")
    print("  - ../../algorithm/DynamicSegmentGraph/buildindex.sh")
    print("  - ../../algorithm/DynamicSegmentGraph/test_range_queries.sh")
    print("=============================================================\n")

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths to shell scripts
    buildindex_script = os.path.join(script_dir, '../../algorithm/DynamicSegmentGraph/buildindex.sh')
    testquery_script = os.path.join(script_dir, '../../algorithm/DynamicSegmentGraph/test_range_queries.sh')

    # Ensure both shell scripts have execute permissions
    subprocess.run(['chmod', '+x', buildindex_script])
    subprocess.run(['chmod', '+x', testquery_script])

    # Run buildindex.sh script
    print(">> Running buildindex.sh ...")
    build_process = subprocess.run([buildindex_script])
    if build_process.returncode != 0:
        print("Error: buildindex.sh failed!")
        return

    print("\n>> buildindex.sh completed successfully.\n")

    # Run test_range_queries.sh script
    print(">> Running test_range_queries.sh ...")
    test_process = subprocess.run([testquery_script])
    if test_process.returncode != 0:
        print("Error: test_range_queries.sh failed!")
        return

    print("\n>> test_range_queries.sh completed successfully.")

if __name__ == "__main__":
    main()
