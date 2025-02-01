#!/usr/bin/env python3
import hashlib
import os
import shutil
import sys


def compute_file_hash(file_path):
    """
    Computes the SHA-256 hash of a file's contents to check uniqueness.
    This ensures uniqueness depends solely on file contents,
    regardless of file name or metadata.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks for efficiency, especially for large files
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_licenses_folder():
    """
    Ensures the current directory has a 'licenses' folder.
    If not, creates it.
    Returns the path to that folder.
    """
    licenses_path = os.path.join(os.getcwd(), "licenses")
    os.makedirs(licenses_path, exist_ok=True)
    return licenses_path


def copy_unique_license(file_path, licenses_path, seen_hashes):
    """
    If the file at file_path has not been seen before (via its hash),
    copy it into the licenses_path folder. The new file name includes the
    original name plus a short part of the hash to prevent collisions.
    """
    file_hash = compute_file_hash(file_path)
    if file_hash not in seen_hashes:
        seen_hashes.add(file_hash)

        # Extract file's base name (e.g., LICENSE.txt) and extension
        base_name = os.path.basename(file_path)
        name_part, _ = os.path.splitext(base_name)

        # Build new file name with part of the hash to avoid name collisions
        new_file_name = f"{name_part}_{file_hash[:8]}"
        new_file_path = os.path.join(licenses_path, new_file_name)

        # Copy the file
        shutil.copy2(file_path, new_file_path)


def find_and_copy_licenses(root_dir):
    """
    Recursively walks through root_dir to find license files with names:
    LICENSE, LICENSE.txt, LICENSE.md, LICENSE.rst.
    Copies each unique license file into 'licenses' and returns the total count.
    """
    target_filenames = {"LICENSE", "LICENSE.txt", "LICENSE.md", "LICENSE.rst"}
    seen_hashes = set()
    licenses_path = ensure_licenses_folder()

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename in target_filenames:
                file_path = os.path.join(dirpath, filename)
                copy_unique_license(file_path, licenses_path, seen_hashes)

    return len(seen_hashes)


def main():
    """
    Usage: python script.py <folder_path>
    Recursively scans <folder_path> for license files, copies unique ones to 'licenses' folder,
    and prints how many unique licenses were found.
    """
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <folder_path>")
        sys.exit(1)

    root_folder = sys.argv[1]

    if not os.path.isdir(root_folder):
        print(f"Error: '{root_folder}' is not a valid directory.")
        sys.exit(1)

    unique_count = find_and_copy_licenses(root_folder)
    print(
        f"Done! Found and copied {unique_count} unique license file(s) into the 'licenses' folder."
    )


if __name__ == "__main__":
    main()
