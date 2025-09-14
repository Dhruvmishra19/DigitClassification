import os 

file_path = "my_document.txt" 
if os.path.exists(file_path): 
    print(f"The file '{file_path}' exists.")
