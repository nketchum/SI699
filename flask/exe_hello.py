def run_hello():
    result = "Hello World"
    filename = "hello.txt"
    with open(filename, "w") as f:
        f.write(result)
    return filename 
