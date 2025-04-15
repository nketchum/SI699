import zipfile
import os

def run_hello():
    result = "Hello World"
    filename = "hello.txt"

    with open(filename, "w") as f:
        f.write(result)
    # return filename 

    zip_filename = "output_bundle.zip"
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        zipf.write("hello.txt")
        # zipf.write("img.png") # keep going
        # zipf.write("path/to/file.txt", arcname="file.txt") # if source files are in a subdir

    return zip_filename
