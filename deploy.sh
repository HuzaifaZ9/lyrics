#!/bin/bash

# Create a temporary directory for deployment
mkdir -p deployment

# Copy necessary files
cp main.py deployment/
cp requirements.txt deployment/

# Create deployment package
cd deployment
pip install -r requirements.txt -t ./
zip -r ../deployment.zip ./*
cd ..

# Clean up
rm -rf deployment

echo "Deployment package created: deployment.zip" 