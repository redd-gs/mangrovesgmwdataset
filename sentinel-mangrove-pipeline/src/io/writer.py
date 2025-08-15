def write_image(image, output_path):
    """Writes the processed image to the specified output path."""
    try:
        Image.fromarray(image).save(output_path)
        print(f"✅ Image saved: {output_path}")
    except Exception as e:
        print(f"❌ Error saving image: {str(e)}")

def write_metadata(metadata, output_path):
    """Writes metadata to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Metadata saved: {output_path}")
    except Exception as e:
        print(f"❌ Error saving metadata: {str(e)}")