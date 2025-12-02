#!/usr/bin/env python3
"""
Utility script to reset Milvus collection
Use this if you're experiencing persistent errors with the collection schema
"""
import os
from dotenv import load_dotenv
from app.milvus_client import drop_collection_if_exists, get_or_create_collection

load_dotenv()

if __name__ == "__main__":
    print("=" * 60)
    print("Milvus Collection Reset Utility")
    print("=" * 60)
    print()
    
    print("⚠ WARNING: This will delete all data in the collection!")
    response = input("Are you sure you want to continue? (yes/no): ")
    
    if response.lower() != "yes":
        print("Cancelled.")
        exit(0)
    
    print()
    print("Dropping existing collection...")
    dropped = drop_collection_if_exists()
    
    if dropped:
        print("Collection dropped successfully.")
    else:
        print("No collection found to drop.")
    
    print()
    print("Creating new collection...")
    collection = get_or_create_collection()
    
    print()
    print("=" * 60)
    print("✓ Collection reset complete!")
    print("=" * 60)
    print("You can now try uploading documents again.")

