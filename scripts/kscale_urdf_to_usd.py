#!/usr/bin/env python3

""" Convenience script to convert KSCALE URDFs to USD format for use in Isaac Sim.

Usage: python3 kscale_urdf_to_usd.py --urdf_path <path_to_urdf> --ext <extension_name>
Example: python3 kscale_urdf_to_usd.py --urdf_path /home/user/.kscale/robots/kbot-v1-naked/robot/kbot-v1-naked.urdf --ext kbot
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def backup_usd_file(file_path: str) -> str:
    """Create a backup of a USD file with timestamp.
    
    Args:
        file_path: Path to the USD file to backup
        
    Returns:
        Path to the backup file
    """
    if not os.path.exists(file_path):
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.replace(".usd", f"_old_{timestamp}.usd")
    shutil.copy2(file_path, backup_path)
    return backup_path


def handle_existing_files(dst_usd: str, dst_meshes: str) -> bool:
    """Handle case where USD files already exist in extension.
    
    Args:
        dst_usd: Path to destination USD file
        dst_meshes: Path to destination instanceable meshes file
        
    Returns:
        True if we should proceed with the copy, False if we should abort
    """
    if not os.path.exists(dst_usd) and not os.path.exists(dst_meshes):
        return True
        
    print("\nWARNING: USD files already exist in the extension:")
    if os.path.exists(dst_usd):
        print(f"- Main USD: {dst_usd}")
    if os.path.exists(dst_meshes):
        print(f"- Props: {dst_meshes}")
        
    while True:
        print("\nHow would you like to proceed?")
        print("1) Overwrite existing files")
        print("2) Create backup and then overwrite")
        print("3) Cancel")
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            return True
        elif choice == "2":
            # Create backups
            backups = []
            if os.path.exists(dst_usd):
                backup = backup_usd_file(dst_usd)
                backups.append(f"Main USD -> {backup}")
            if os.path.exists(dst_meshes):
                backup = backup_usd_file(dst_meshes)
                backups.append(f"Props -> {backup}")
            print("\nCreated backups:")
            for backup in backups:
                print(f"- {backup}")
            return True
        elif choice == "3":
            print("\nOperation cancelled.")
            return False
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")


def convert_kscale_robot(urdf_path: str, ext_name: str, fix_base: bool = False, merge_joints: bool = False):
    """Convert a KSCALE robot's URDF to USD format.
    
    Args:
        urdf_path: Full path to the URDF file
        ext_name: Name of the extension to copy USD files to (e.g. kbot)
        fix_base: Whether to fix the base link to where it is imported
        merge_joints: Whether to consolidate links connected by fixed joints
    """
    # Check if URDF exists
    if not os.path.exists(urdf_path):
        print(f"Error: URDF not found at {urdf_path}")
        sys.exit(1)

    # Get klab directory from script location
    klab_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Setup USD output directory (next to URDF)
    kscale_usd_dir = os.path.join(os.path.dirname(urdf_path), "usd")
    os.makedirs(kscale_usd_dir, exist_ok=True)

    # Get robot name from URDF path
    urdf_name = os.path.splitext(os.path.basename(urdf_path))[0]

    # Convert URDF to USD using Isaac Sim's standalone converter
    print(f"Converting {urdf_path} to USD...")
    isaac_lab_path = os.environ.get("ISAAC_LAB_PATH")
    if not isaac_lab_path:
        print("Error: ISAAC_LAB_PATH environment variable not set")
        sys.exit(1)
        
    converter_script = os.path.join(isaac_lab_path, "source", "standalone", "tools", "convert_urdf.py")
    cmd = [
        os.path.join(isaac_lab_path, "isaaclab.sh"),
        "-p", converter_script,
        urdf_path,  # input (positional)
        os.path.join(kscale_usd_dir, f"{urdf_name}.usd"),  # output (positional)
        "--make-instanceable",  # we want instanceable assets
        "--headless",  # run without GUI
    ]
    
    # Add optional flags
    if fix_base:
        cmd.append("--fix-base")
    if merge_joints:
        cmd.append("--merge-joints")
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running URDF converter: {e}")
        sys.exit(1)

    print(f"Successfully converted to:\n{os.path.join(kscale_usd_dir, urdf_name + '.usd')}")

    # Check if extension exists
    ext_dir = os.path.join(klab_dir, "exts", ext_name)
    if not os.path.exists(ext_dir):
        print(f"Error: Extension directory not found at {ext_dir}")
        print("Please create the extension first using the template.")
        sys.exit(1)

    # Setup extension paths
    ext_props_dir = os.path.join(ext_dir, ext_name, "assets", "Robots", "Props_test")
    ext_robots_dir = os.path.join(ext_dir, ext_name, "assets", "Robots")
    
    # Ensure Props directory exists
    if not os.path.exists(ext_props_dir):
        print(f"Error: Props_test directory not found at {ext_props_dir}")
        print("Please make sure the extension has the correct directory structure.")
        sys.exit(1)

    # Prepare destination paths
    # Check both possible locations for instanceable meshes
    src_meshes = os.path.join(kscale_usd_dir, "Props", "instanceable_meshes.usd")
    if not os.path.exists(src_meshes):
        src_meshes = os.path.join(kscale_usd_dir, "instanceable_meshes.usd")
        if not os.path.exists(src_meshes):
            print(f"Error: Could not find instanceable meshes file in either:")
            print(f"- {os.path.join(kscale_usd_dir, 'Props', 'instanceable_meshes.usd')}")
            print(f"- {os.path.join(kscale_usd_dir, 'instanceable_meshes.usd')}")
            sys.exit(1)

    dst_meshes = os.path.join(ext_props_dir, "instanceable_meshes.usd")
    src_usd = os.path.join(kscale_usd_dir, f"{urdf_name}.usd")
    dst_usd = os.path.join(ext_robots_dir, f"{ext_name}.usd")

    # Check if files exist and handle accordingly
    if not handle_existing_files(dst_usd, dst_meshes):
        sys.exit(0)

    # Copy files
    shutil.copy2(src_meshes, dst_meshes)
    shutil.copy2(src_usd, dst_usd)

    print(f"\nCopied and renamed USD files to extension:")
    print(f"Main USD: {dst_usd}")
    print(f"Props: {dst_meshes}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Convert KSCALE robot URDF to USD format")
    parser.add_argument("--urdf_path", type=str, required=True, 
                      help="Full path to the URDF file")
    parser.add_argument("--ext", type=str, required=True,
                      help="Name of the extension to copy USD files to (e.g. kbot)")
    parser.add_argument("--fix-base", action="store_true",
                      help="Fix the base link to where it is imported")
    parser.add_argument("--merge-joints", action="store_true",
                      help="Consolidate links that are connected by fixed joints")
    args = parser.parse_args()

    # Convert the robot
    convert_kscale_robot(args.urdf_path, args.ext, args.fix_base, args.merge_joints)


if __name__ == "__main__":
    main()

