import os
import subprocess

def main():
    # Display reminder message
    print("Please make sure the IrangeData dataset has been placed in the /HybridANNS/data/temp directory before running.")
    input("Press Enter to continue...")

    # Build the relative path to exp.sh
    script_path = os.path.join('..', '..', 'algorithm', 'iRangeGraph', 'exp.sh')
    
    # Ensure the script has execute permission
    subprocess.run(['chmod', '+x', script_path], check=True)
    
    # Run the shell script
    subprocess.run([script_path], check=True)

if __name__ == "__main__":
    main()
