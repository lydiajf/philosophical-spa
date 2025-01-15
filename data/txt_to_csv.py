import csv
import re

def process_philosophical_texts(input_file: str, output_file: str):
    """Process philosophical texts from txt to csv format"""
    with open(input_file, "r", encoding='utf-8') as file:
        lines = file.readlines()

    # Initialize variables
    data = []
    current_framework = None
    current_author = None
    current_work = None
    current_focus = None
    current_excerpt = []
    
    for line in lines:
        line = line.strip()
        
        if not line:  # Skip empty lines
            continue
            
        if line == "---":  # Reset all variables at section separator
            if current_excerpt:
                data.append([
                    current_framework,
                    current_author,
                    current_work,
                    current_focus,
                    " ".join(current_excerpt)
                ])
            # Reset all variables for next entry
            current_framework = None
            current_author = None
            current_work = None
            current_focus = None
            current_excerpt = []
            continue
            
        # Check if this is a framework line
        if not current_framework:
            current_framework = line
            continue
            
        # Check if this is an author line
        if not current_author:
            current_author = line
            continue
            
        # Check if this is a work line
        if not current_work:
            current_work = line
            continue
            
        # Check if this is a focus line
        if not current_focus:
            current_focus = line
            continue
            
        # If we have all metadata, this must be part of the excerpt
        current_excerpt.append(line)
    
    # Don't forget to save the last entry if there is one
    if current_excerpt:
        data.append([
            current_framework,
            current_author,
            current_work,
            current_focus,
            " ".join(current_excerpt)
        ])

    # Write to CSV
    with open(output_file, "w", newline="", encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Ethical Framework", "Author", "Work", "Focus", "Excerpt"])
        writer.writerows(data)

    print(f"Created CSV with {len(data)} philosophical excerpts")
    
    # Print preview of the data
    print("\nFirst entry preview:")
    for field, value in zip(["Framework", "Author", "Work", "Focus"], data[0][:4]):
        print(f"{field}: {value}")
    print(f"Excerpt length: {len(data[0][4])} characters")

if __name__ == "__main__":
    process_philosophical_texts("data/philosophers.txt", "philosophical_excerpts.csv")
