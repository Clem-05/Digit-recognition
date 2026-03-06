DATABASE = {
    "0": [1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1],
    "1": [0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0],
    "2": [1,1,1,1,1, 0,0,0,0,1, 1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,1]
}

def get_image_data(file_path):
    """Acquires, cleans, and flattens image data from a PBM file."""
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        raw_pixels = "".join(lines[2:]).split()
        return [int(p) for p in raw_pixels]
    except Exception as e:
        print(f"Error: {e}")
        return None

def predict(input_data, database):
    """Compares input list to database templates using linear matching."""
    best_digit = None
    max_score = -1
    
    for digit, template in database.items():
        matches = sum(1 for p1, p2 in zip(input_data, template) if p1 == p2)
        
        if matches > max_score:
            max_score = matches
            best_digit = digit
            
    confidence = (max_score / len(input_data)) * 100
    return best_digit, confidence

FILENAME = "test_digit.pbm"
image_data = get_image_data(FILENAME)

if image_data:
    result, score = predict(image_data, DATABASE)
    print(f"File: {FILENAME}")
    print(f"AI Prediction: {result}")
    print(f"Confidence: {score:.1f}%")

def print_matrix(data):
    """Prints the 5x5 matrix in the console for debugging"""
    for i in range(0, 25, 5):
        row = data[i:i+5]
        visual_row = "".join(["■ " if p == 1 else ". " for p in row])
        print(visual_row)

print("Image analyzed by AI:")
print_matrix(image_data)