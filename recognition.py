REFERENCE_DATABASE = {
    "0": [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ],
    "1": [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0]
    ],
    "2": [
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1]
    ]
}

def process_image(file_path):
    """
    Steps 2 to 5: Simulated processing of a PBM file.
    PBM (Portable BitMap) is a text-based image format.
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        raw_data = "".join(lines[2:]).replace("\n", "").replace(" ", "")
       
        matrix = []
        for i in range(0, len(raw_data), 5):
            row = [int(pixel) for pixel in raw_data[i:i+5]]
            matrix.append(row)
        return matrix
    except Exception as e:
        print(f"Error during acquisition: {e}")
        return None

def compare_to_database(input_matrix, database):
    best_score = -1
    recognized_digit = None
    total_pixels = 25 # 5x5 grid

    for digit, template in database.items():
        current_score = 0
        for y in range(5):
            for x in range(5):
                # Pixel matching logic
                if input_matrix[y][x] == template[y][x]:
                    current_score += 1
        
        if current_score > best_score:
            best_score = current_score
            recognized_digit = digit
            
    confidence = (best_score / total_pixels) * 100
    return recognized_digit, confidence


test_image = [
    [0, 1, 1, 0, 0],
    [1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 1, 1, 0]
]

result, accuracy = compare_to_database(test_image, REFERENCE_DATABASE)

print("--- AI Digit Recognition ---")
print(f"Input image processed successfully.")
print(f"Prediction: This digit is a '{result}'")
print(f"Confidence Level: {accuracy}%")